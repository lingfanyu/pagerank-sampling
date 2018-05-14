#include <cstdio>
#include <cstdlib>
#include <string>
#include <pthread.h>
#include <vector>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <functional>

#define NTHREAD 32
#define PERCENT 0.01
#define NSAMPLE 1024

using namespace std;
using Edges = vector<pair<int, int> >;
using Sampler = function<unordered_set<int>(int, const Edges&, int)>;

unordered_set<int> bfs_sampler(int num_vertex, const Edges& edges, int n_sample) {
    // build an adj matrix
    unordered_map<int, vector<int>> adj;
    for (auto& e: edges) {
        if (adj.find(e.first) == adj.end()) {
            adj[e.first] = vector<int>();
        }
        adj[e.first].push_back(e.second);
    }

    unordered_set<int> res;

    // bfs
    vector<int> queue;

    // choose random start point
    int start = rand() % num_vertex;
    res.insert(start);
    queue.push_back(start);

    int pos = 0;
    int count = 1;
    while (pos < queue.size() && count < n_sample) {
        int cur = queue[pos++];
        auto& es = adj[cur];
        for (int dst: es) {
            if (res.find(dst) == res.end()) {
                res.insert(dst);
                queue.push_back(dst);
                ++count;
            }
            if (count >= n_sample) {
                break;
            }
        }
    }

    return move(res);
}

unordered_set<int> uniform_sampler(int num_vertex, const Edges& edges, int n_sample) {
    unordered_set<int> res;
    while (res.size() < n_sample) {
        res.insert(rand() % num_vertex);
    }
    return move(res);
}

void dump(string file, const Edges& edges, unordered_set<int>& v) {
    Edges new_edges;
    for (auto& e: edges) {
        if (v.find(e.first) != v.end() && v.find(e.second) != v.end()) {
            new_edges.push_back(e);
        }
    }
    FILE* fp = fopen(file.c_str(), "w");
    fprintf(fp, "#\n#\n# Nodes: %lu Edges: %lu\n#\n", v.size(), new_edges.size());
    for (auto& e: new_edges) {
        fprintf(fp, "%d\t%d\n", e.first, e.second);
    }
    fclose(fp);
}

struct Arg{
    const Edges& edges;
    Sampler& sampler;
    vector<string> files;
    int num_vertex;
    int n_sample;
    Arg(const Edges& a1, Sampler& a2, vector<string>& a3, int a4, int a5) :
        edges(a1), sampler(a2), files(a3), num_vertex(a4), n_sample(a5) {}
};

void* sampler_worker(void* args) {
    Arg* a = (Arg*)args;
    for (auto& file: a->files) {
        auto vertices = a->sampler(a->num_vertex, a->edges, a->n_sample);
        dump(file, a->edges, vertices);
    }
    return NULL;
}

int main() {
    srand(time(NULL));
    string path = "samples";
    Sampler sampler = uniform_sampler;
    const char* filename = "web-Stanford.txt";
    FILE* fp = fopen(filename, "r");
    char buf[128];
    fgets(buf, 128, fp);
    fgets(buf, 128, fp);
    fgets(buf, 128, fp);
    int n_v, n_e;
    sscanf(buf, "# Nodes: %d Edges: %d", &n_v, &n_e);
    fgets(buf, 128, fp);
    printf("%d %d\n", n_v, n_e);
    Edges edges;
    for (int i = 0; i < n_e; i++) {
        int src, dst;
        fscanf(fp, "%d\t%d\n", &src, &dst);
        edges.emplace_back(src, dst);
        edges.emplace_back(dst, src);
    }
    fclose(fp);

    int n_sample = int(n_v * PERCENT);
    vector<Arg> args;
    int each = NSAMPLE / NTHREAD * 8;
    string prefix = path + "/sample_";
    for (int i = 0; i < NTHREAD; i++) {
        vector<string> files;
        for (int j = 0; j < each; j++) {
            files.push_back(prefix + to_string(i * each + j) + ".txt");
        }
        args.emplace_back(edges, sampler, files, n_v, n_sample);
    }

    vector<pthread_t> pids;
    for (int i = 0; i < NTHREAD; i++) {
        pthread_t pid;
        pthread_create(&pid, NULL, sampler_worker, (void*)&(args[i]));
        pids.push_back(pid);
    }

    for (auto& pid: pids) {
        pthread_join(pid, NULL);
    }

    return 0;
}
