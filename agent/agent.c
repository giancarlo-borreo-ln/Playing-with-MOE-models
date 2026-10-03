/*
 * agent.c - a tiny coding agent for a local MoE model (qwen3:30b-a3b).
 *
 * The agent talks to an Ollama server over a raw POSIX socket (no libcurl,
 * no JSON library), keeps a multi-turn conversation, extracts the fenced
 * code block the model writes, and runs it inside a throwaway Docker
 * container with no network, a read-only root, capped CPU/RAM/PIDs and a
 * wall-clock timeout. The program's output is fed back to the model so it
 * can fix its own mistakes until it reports DONE.
 *
 * Build:  make            (see Makefile)
 * Usage:  ./agent "write a C program that prints the first 20 primes"
 *
 * Config via environment:
 *   OLLAMA_HOST     default 127.0.0.1
 *   OLLAMA_PORT     default 11434
 *   MODEL           default qwen3:30b-a3b
 *   MAX_TURNS       default 8       conversation rounds
 *   LLM_TIMEOUT     default 900     seconds to wait for the model
 *   RUN_TIMEOUT     default 10      seconds allowed for generated code
 *   SANDBOX_IMAGE   default agent-sandbox:latest
 *   SANDBOX_MEM     default 256m
 *   SANDBOX_CPUS    default 1
 *   TEMPERATURE     default 0.2
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ buffer */
typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} Buf;

static void buf_init(Buf *b) {
    b->cap = 1024;
    b->len = 0;
    b->data = malloc(b->cap);
    b->data[0] = '\0';
}

static void buf_reserve(Buf *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    while (b->len + extra + 1 > b->cap) b->cap *= 2;
    b->data = realloc(b->data, b->cap);
}

static void buf_append(Buf *b, const char *s, size_t n) {
    buf_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void buf_puts(Buf *b, const char *s) { buf_append(b, s, strlen(s)); }

static void buf_putc(Buf *b, char c) { buf_append(b, &c, 1); }

static void buf_printf(Buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    buf_reserve(b, (size_t)n);
    va_start(ap, fmt);
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    b->len += (size_t)n;
}

static void buf_free(Buf *b) { free(b->data); }

/* --------------------------------------------------------------- json text */
/* Append s to b as the *contents* of a JSON string (escapes added). */
static void json_escape(Buf *b, const char *s) {
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"':  buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\n': buf_puts(b, "\\n");  break;
        case '\r': buf_puts(b, "\\r");  break;
        case '\t': buf_puts(b, "\\t");  break;
        case '\b': buf_puts(b, "\\b");  break;
        case '\f': buf_puts(b, "\\f");  break;
        default:
            if (*p < 0x20) buf_printf(b, "\\u%04x", *p);
            else buf_putc(b, (char)*p);
        }
    }
}

/* Decode a JSON string body [s,s+len) (without surrounding quotes). */
static char *json_unescape(const char *s, size_t len) {
    Buf out;
    buf_init(&out);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c != '\\') { buf_putc(&out, (char)c); continue; }
        if (++i >= len) break;
        switch (s[i]) {
        case '"':  buf_putc(&out, '"');  break;
        case '\\': buf_putc(&out, '\\'); break;
        case '/':  buf_putc(&out, '/');  break;
        case 'n':  buf_putc(&out, '\n'); break;
        case 'r':  buf_putc(&out, '\r'); break;
        case 't':  buf_putc(&out, '\t'); break;
        case 'b':  buf_putc(&out, '\b'); break;
        case 'f':  buf_putc(&out, '\f'); break;
        case 'u': {
            if (i + 4 >= len) { i = len; break; }
            char hex[5] = { s[i+1], s[i+2], s[i+3], s[i+4], 0 };
            unsigned cp = (unsigned)strtoul(hex, NULL, 16);
            i += 4;
            if (cp < 0x80) {
                buf_putc(&out, (char)cp);
            } else if (cp < 0x800) {
                buf_putc(&out, (char)(0xC0 | (cp >> 6)));
                buf_putc(&out, (char)(0x80 | (cp & 0x3F)));
            } else {
                buf_putc(&out, (char)(0xE0 | (cp >> 12)));
                buf_putc(&out, (char)(0x80 | ((cp >> 6) & 0x3F)));
                buf_putc(&out, (char)(0x80 | (cp & 0x3F)));
            }
            break;
        }
        default: buf_putc(&out, s[i]); break;
        }
    }
    return out.data;
}

/* Find "key": "value" in a JSON object and return the decoded value. */
static char *json_get_string(const char *json, const char *key) {
    size_t klen = strlen(key);
    char *pat = malloc(klen + 3);
    sprintf(pat, "\"%s\"", key);

    char *found = NULL;
    const char *p = json;
    while ((p = strstr(p, pat)) != NULL) {
        const char *q = p + klen + 2;
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q == ':') {
            q++;
            while (*q && isspace((unsigned char)*q)) q++;
            if (*q == '"') {
                q++;
                const char *end = q;
                while (*end) {
                    if (*end == '\\' && end[1]) { end += 2; continue; }
                    if (*end == '"') break;
                    end++;
                }
                found = json_unescape(q, (size_t)(end - q));
                break;
            }
        }
        p++;
    }
    free(pat);
    return found;
}

/* ------------------------------------------------------------------- http */
static int http_post(const char *host, const char *port, const char *path,
                     const char *body, int timeout_s, Buf *resp) {
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, port, &hints, &res) != 0 || !res) {
        fprintf(stderr, "agent: cannot resolve %s:%s\n", host, port);
        return -1;
    }

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        fprintf(stderr, "agent: cannot connect to %s:%s (%s)\n",
                host, port, strerror(errno));
        return -1;
    }

    Buf req;
    buf_init(&req);
    buf_printf(&req,
        "POST %s HTTP/1.1\r\n"
        "Host: %s:%s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        path, host, port, strlen(body));
    buf_puts(&req, body);

    size_t sent = 0;
    while (sent < req.len) {
        ssize_t n = send(fd, req.data + sent, req.len - sent, MSG_NOSIGNAL);
        if (n <= 0) { fprintf(stderr, "agent: send failed: %s\n", strerror(errno)); close(fd); buf_free(&req); return -1; }
        sent += (size_t)n;
    }
    buf_free(&req);

    /* Read the whole response (Connection: close). */
    Buf raw;
    buf_init(&raw);
    time_t deadline = time(NULL) + timeout_s;
    char tmp[65536];
    for (;;) {
        int remain_ms = (int)((deadline - time(NULL)) * 1000);
        if (remain_ms <= 0) { fprintf(stderr, "agent: model response timed out\n"); close(fd); buf_free(&raw); return -1; }
        struct pollfd pfd = { .fd = fd, .events = POLLIN };
        int pr = poll(&pfd, 1, remain_ms);
        if (pr == 0) { fprintf(stderr, "agent: model response timed out\n"); close(fd); buf_free(&raw); return -1; }
        if (pr < 0) { if (errno == EINTR) continue; break; }
        ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n < 0) { if (errno == EINTR) continue; break; }
        if (n == 0) break;
        buf_append(&raw, tmp, (size_t)n);
    }
    close(fd);

    char *hdr_end = strstr(raw.data, "\r\n\r\n");
    if (!hdr_end) { fprintf(stderr, "agent: malformed HTTP response\n"); buf_free(&raw); return -1; }
    int status = 0;
    sscanf(raw.data, "HTTP/%*s %d", &status);
    char *bstart = hdr_end + 4;
    size_t blen = raw.len - (size_t)(bstart - raw.data);

    /* Handle chunked transfer-encoding. */
    if (strcasestr(raw.data, "Transfer-Encoding: chunked")) {
        char *p = bstart;
        char *end = raw.data + raw.len;
        Buf body;
        buf_init(&body);
        while (p < end) {
            char *nl = memchr(p, '\n', (size_t)(end - p));
            if (!nl) break;
            long sz = strtol(p, NULL, 16);
            if (sz <= 0) break;
            char *chunk = nl + 1;
            if (chunk + sz > end) sz = (long)(end - chunk);
            buf_append(&body, chunk, (size_t)sz);
            p = chunk + sz;
            if (p + 1 < end && p[0] == '\r' && p[1] == '\n') p += 2;
        }
        buf_free(&raw);
        if (body.len == 0) { buf_free(&body); return -1; }
        resp->data = body.data;
        resp->len = body.len;
        resp->cap = body.cap;
    } else {
        Buf body;
        buf_init(&body);
        buf_append(&body, bstart, blen);
        buf_free(&raw);
        resp->data = body.data;
        resp->len = body.len;
        resp->cap = body.cap;
    }

    if (status != 200) {
        fprintf(stderr, "agent: HTTP %d from server\n", status);
        return -1;
    }
    return 0;
}

/* -------------------------------------------------------- code extraction */
typedef struct {
    char *lang;  /* malloc'd, may be "" */
    char *code;  /* malloc'd */
} CodeBlock;

/* Extract the first ``` fenced block. Returns 0 if one was found. */
static int extract_code(const char *text, CodeBlock *out) {
    const char *open = strstr(text, "```");
    if (!open) return -1;

    const char *info = open + 3;
    const char *nl = strchr(info, '\n');
    if (!nl) return -1;

    size_t langlen = (size_t)(nl - info);
    while (langlen && (info[langlen - 1] == '\r' || info[langlen - 1] == ' '))
        langlen--;
    out->lang = malloc(langlen + 1);
    memcpy(out->lang, info, langlen);
    out->lang[langlen] = '\0';

    const char *start = nl + 1;
    const char *close = strstr(start, "```");
    if (!close) return -1;
    size_t len = (size_t)(close - start);
    out->code = malloc(len + 1);
    memcpy(out->code, start, len);
    out->code[len] = '\0';
    return 0;
}

/* ------------------------------------------------------------- docker run */
static int run_child(char **argv, int timeout_s, Buf *out) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return -1; }

    if (pid == 0) {
        setpgid(0, 0);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
        execvp(argv[0], argv);
        fprintf(stderr, "exec %s: %s\n", argv[0], strerror(errno));
        _exit(127);
    }

    close(pipefd[1]);
    setpgid(pid, pid);

    time_t deadline = time(NULL) + timeout_s;
    int timed_out = 0;
    char tmp[8192];
    for (;;) {
        int remain_ms = (int)((deadline - time(NULL)) * 1000);
        if (remain_ms <= 0) {
            timed_out = 1;
            kill(-pid, SIGKILL);
            break;
        }
        struct pollfd pfd = { .fd = pipefd[0], .events = POLLIN };
        int pr = poll(&pfd, 1, remain_ms);
        if (pr == 0) { timed_out = 1; kill(-pid, SIGKILL); break; }
        if (pr < 0) { if (errno == EINTR) continue; break; }
        ssize_t n = read(pipefd[0], tmp, sizeof tmp);
        if (n < 0) { if (errno == EINTR) continue; break; }
        if (n == 0) break;
        buf_append(out, tmp, (size_t)n);
    }
    close(pipefd[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    if (timed_out) {
        buf_printf(out, "\n[sandbox] killed after %ds timeout\n", timeout_s);
        return 124;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

static void docker_cleanup(const char *name) {
    pid_t pid = fork();
    if (pid == 0) {
        execlp("docker", "docker", "rm", "-f", name, (char *)NULL);
        _exit(127);
    }
    if (pid > 0) waitpid(pid, NULL, 0);
}

struct LangMap { const char *name; const char *file; const char *cmd; };
static const struct LangMap LMAP[] = {
    { "c",        "main.c",   "cc -O2 -Wall -Wextra -o prog main.c && ./prog" },
    { "cpp",      "main.cpp", "c++ -O2 -Wall -o prog main.cpp && ./prog" },
    { "c++",      "main.cpp", "c++ -O2 -Wall -o prog main.cpp && ./prog" },
    { "python",   "main.py",  "python3 main.py" },
    { "python3",  "main.py",  "python3 main.py" },
    { "py",       "main.py",  "python3 main.py" },
    { "bash",     "main.sh",  "bash main.sh" },
    { "sh",       "main.sh",  "sh main.sh" },
    { "shell",    "main.sh",  "bash main.sh" },
    { NULL, NULL, NULL }
};

static const struct LangMap *find_lang(const char *lang) {
    if (!lang) return NULL;
    for (const struct LangMap *l = LMAP; l->name; l++) {
        if (strcasecmp(l->name, lang) == 0) return l;
    }
    return NULL;
}

/*
 * Write `code` into a fresh temp dir and execute it in a locked-down
 * container. Returns the combined output via `out` (caller frees out->data).
 * ok_out is set to 1 when the container exited 0.
 */
static int run_sandbox(const char *image, const char *mem, const char *cpus,
                       const CodeBlock *cb, int timeout_s, Buf *out, int *ok_out) {
    *ok_out = 0;

    const struct LangMap *lm = find_lang(cb->lang);
    if (!lm) {
        buf_printf(out, "[sandbox] unsupported or missing language tag \"%s\"; "
                        "use c, cpp, python or bash.\n", cb->lang);
        return -1;
    }

    char tmpl[] = "/tmp/agent-sandbox-XXXXXX";
    char *dir = mkdtemp(tmpl);
    if (!dir) { fprintf(stderr, "agent: mkdtemp: %s\n", strerror(errno)); return -1; }
    chmod(dir, 0777);

    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, lm->file);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "agent: cannot write %s: %s\n", path, strerror(errno)); return -1; }
    fputs(cb->code, f);
    fclose(f);
    chmod(path, 0644);

    char mount[600];
    snprintf(mount, sizeof mount, "%s:/work:rw", dir);

    static unsigned run_id = 0;
    char name[64];
    snprintf(name, sizeof name, "agent-run-%d-%u", (int)getpid(), ++run_id);

    char *argv[] = {
        "docker", "run", "--rm",
        "--name", name,
        "--network", "none",
        "--memory", (char *)mem,
        "--memory-swap", (char *)mem,
        "--cpus", (char *)cpus,
        "--pids-limit", "128",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "-e", "HOME=/tmp",
        "-v", mount,
        "-w", "/work",
        (char *)image,
        "sh", "-c", (char *)lm->cmd,
        NULL
    };

    buf_printf(out, "$ %s\n", lm->cmd);
    int rc = run_child(argv, timeout_s, out);
    if (rc == 124) docker_cleanup(name);

    /* Keep the sandbox dir for inspection on failure; otherwise remove it. */
    if (rc == 0) {
        char rm[600];
        snprintf(rm, sizeof rm, "%s/%s", dir, lm->file);
        unlink(rm);
        char prog[600];
        snprintf(prog, sizeof prog, "%s/prog", dir);
        unlink(prog);
        rmdir(dir);
        *ok_out = 1;
    } else {
        fprintf(stderr, "agent: sandbox files kept at %s\n", dir);
    }
    return rc;
}

/* --------------------------------------------------------------- messages */
typedef struct { char *role; char *content; } Msg;

typedef struct {
    Msg   *v;
    int    n;
    int    cap;
} Msgs;

static void msgs_push(Msgs *m, const char *role, const char *content) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 8;
        m->v = realloc(m->v, (size_t)m->cap * sizeof(Msg));
    }
    m->v[m->n].role = strdup(role);
    m->v[m->n].content = strdup(content);
    m->n++;
}

static void build_request(const char *model, const Msgs *m, double temp, int ctx, Buf *out) {
    buf_puts(out, "{\"model\":\"");
    json_escape(out, model);
    buf_puts(out, "\",\"messages\":[");
    for (int i = 0; i < m->n; i++) {
        if (i) buf_putc(out, ',');
        buf_puts(out, "{\"role\":\"");
        json_escape(out, m->v[i].role);
        buf_puts(out, "\",\"content\":\"");
        json_escape(out, m->v[i].content);
        buf_puts(out, "\"}");
    }
    buf_printf(out,
        "],\"stream\":false,\"think\":false,"
        "\"options\":{\"temperature\":%.2f,\"num_ctx\":%d}}",
        temp, ctx);
}

/* ---------------------------------------------------------------- helpers */
static const char *env_or(const char *k, const char *d) {
    const char *v = getenv(k);
    return (v && *v) ? v : d;
}

static const char *SYSTEM_PROMPT =
    "You are an autonomous coding agent running on a Linux box. "
    "You solve the user's task by writing a program and running it.\n"
    "RULES:\n"
    "1. Reply with exactly ONE fenced code block containing a complete program. "
    "Tag it as c, cpp, python or bash.\n"
    "2. The sandbox has gcc, g++, python3 and bash, but NO network access, "
    "256 MB of RAM, 1 CPU and a 10 second timeout. Use only the standard "
    "library; do not read or write files outside the working directory.\n"
    "3. The program must print its result to stdout.\n"
    "4. After each run you will receive the program's stdout/stderr. If it "
    "failed or is wrong, reply with a corrected code block.\n"
    "5. When the task is complete and verified, reply with a single line that "
    "starts with DONE: followed by a short summary, and no code block.\n"
    "Start by briefly stating your plan in one sentence, then the code block.";

/* --------------------------------------------------------------- the loop */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s \"task description\"\n"
            "\n"
            "A minimal C coding agent that sends the task to a local Ollama\n"
            "model and runs the code it writes inside a locked-down container.\n"
            "\n"
            "env: MODEL (qwen3:30b-a3b) OLLAMA_HOST (127.0.0.1) OLLAMA_PORT (11434)\n"
            "     MAX_TURNS (8) LLM_TIMEOUT (900) RUN_TIMEOUT (10)\n"
            "     SANDBOX_IMAGE (agent-sandbox:latest) SANDBOX_MEM (256m)\n"
            "     SANDBOX_CPUS (1) TEMPERATURE (0.2)\n",
            argv[0]);
        return 2;
    }

    Buf task;
    buf_init(&task);
    for (int i = 1; i < argc; i++) {
        if (i > 1) buf_putc(&task, ' ');
        buf_puts(&task, argv[i]);
    }

    const char *host    = env_or("OLLAMA_HOST", "127.0.0.1");
    const char *port    = env_or("OLLAMA_PORT", "11434");
    const char *model   = env_or("MODEL", "qwen3:30b-a3b");
    const char *image   = env_or("SANDBOX_IMAGE", "agent-sandbox:latest");
    const char *mem     = env_or("SANDBOX_MEM", "256m");
    const char *cpus    = env_or("SANDBOX_CPUS", "1");
    int max_turns       = atoi(env_or("MAX_TURNS", "8"));
    int llm_timeout     = atoi(env_or("LLM_TIMEOUT", "900"));
    int run_timeout     = atoi(env_or("RUN_TIMEOUT", "10"));
    double temperature  = atof(env_or("TEMPERATURE", "0.2"));

    printf("\033[1;36m[agent]\033[0m model=%s host=%s:%s image=%s turns=%d\n",
           model, host, port, image, max_turns);
    printf("\033[1;36m[agent]\033[0m task: %s\n", task.data);

    Msgs msgs = {0};
    msgs_push(&msgs, "system", SYSTEM_PROMPT);
    msgs_push(&msgs, "user", task.data);

    int final_rc = 0;

    for (int turn = 1; turn <= max_turns; turn++) {
        Buf req;
        buf_init(&req);
        build_request(model, &msgs, temperature, 8192, &req);

        printf("\n\033[1;36m[agent]\033[0m thinking (turn %d/%d) ...\n", turn, max_turns);
        fflush(stdout);

        Buf resp;
        buf_init(&resp);
        if (http_post(host, port, "/api/chat", req.data, llm_timeout, &resp) != 0) {
            buf_free(&req);
            buf_free(&resp);
            final_rc = 1;
            break;
        }
        buf_free(&req);

        char *content = json_get_string(resp.data, "content");
        buf_free(&resp);
        if (!content) {
            fprintf(stderr, "agent: no 'content' in model reply\n");
            final_rc = 1;
            break;
        }

        printf("\n\033[1;35m--- model ---\033[0m\n%s\n", content);

        CodeBlock cb = {0};
        int have_code = (extract_code(content, &cb) == 0);

        if (!have_code) {
            if (strstr(content, "DONE")) {
                printf("\n\033[1;32m[agent] model reported DONE.\033[0m\n");
                free(content);
                break;
            }
            msgs_push(&msgs, "assistant", content);
            msgs_push(&msgs, "user",
                "I did not find a runnable code block. Reply with exactly one "
                "fenced code block (c, cpp, python or bash), or DONE: <summary>.");
            free(content);
            continue;
        }

        printf("\n\033[1;35m--- code (%s) ---\033[0m\n%s\n",
               cb.lang[0] ? cb.lang : "?", cb.code);

        Buf runout;
        buf_init(&runout);
        int ok = 0;
        printf("\n\033[1;36m[agent]\033[0m running in sandbox ...\n");
        int rc = run_sandbox(image, mem, cpus, &cb, run_timeout, &runout, &ok);
        printf("\033[1;35m--- output (exit %d) ---\033[0m\n%s\n", rc, runout.data);

        size_t limit = 4000;
        if (runout.len > limit) {
            runout.len = limit;
            runout.data[limit] = '\0';
            buf_puts(&runout, "\n...[output truncated]");
        }

        msgs_push(&msgs, "assistant", content);

        Buf feedback;
        buf_init(&feedback);
        buf_printf(&feedback,
            "Program exited with status %d. Output:\n```\n%s\n```\n", rc, runout.data);
        if (ok)
            buf_puts(&feedback,
                "That looks successful. If the output fully satisfies the task, "
                "reply with `DONE: <summary>` and no code. Otherwise send a "
                "corrected code block.");
        else
            buf_puts(&feedback,
                "Fix the code and reply with a corrected code block in the same "
                "fenced format.");

        msgs_push(&msgs, "user", feedback.data);

        buf_free(&feedback);
        buf_free(&runout);
        free(cb.lang);
        free(cb.code);
        free(content);
    }

    for (int i = 0; i < msgs.n; i++) {
        free(msgs.v[i].role);
        free(msgs.v[i].content);
    }
    free(msgs.v);
    buf_free(&task);
    return final_rc;
}
