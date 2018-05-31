#include <cstdio>
#include <cstdlib>
#include <string>
#include <pthread.h>
#include <vector>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <algorithm>

#define NTHREAD 32
#define PERCENT 0.85
#define NSAMPLE 256

using namespace std;
using Edges = vector<pair<int, int> >;
using Sampler = function<unordered_set<int>(int, const Edges&, int)>;
#ifdef MIN_NEIGHBOR
using VCount = unordered_map<int, int>;
pthread_mutex_t mutex;
VCount vc;
class ScopeLock {
public:
    ScopeLock() {
        pthread_mutex_lock(&mutex);
    }
    ~ScopeLock() {
        pthread_mutex_unlock(&mutex);
    }
};
#endif

struct pairhash {
public:
    std::size_t operator()(const std::pair<int, int> &x) const{
        return std::hash<string>()(to_string(x.first) + "," + to_string(x.second));
    }
};

bool my_comparer(const pair<int, int>& x, const pair<int, int>& y) {
    return x.first < y.first || x.first == y.first && x.second <= y.second;
}

bool count_comparer(const pair<int, int>& x, const pair<int, int>& y) {
    return x.second < y.second;
}

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
    // original graph is 1-indexed
    int start = rand() % num_vertex + 1;
    res.insert(start);
    queue.push_back(start);

    int pos = 0;
    int count = 1;
    while (pos < queue.size() && count < n_sample) {
        int cur = queue[pos++];
#ifdef MIN_NEIGHBOR
        auto& ess = adj[cur];
        Edges vcount;
        {
            ScopeLock l;
            for (int dst: ess) {
                vcount.emplace_back(dst, vc[dst]);
            }
        }
        sort(vcount.begin(), vcount.end(), count_comparer);
        vector<int> es;
        for (auto& en: vcount) {
            es.push_back(en.first);
        }
#else
        auto& es = adj[cur];
        random_shuffle(es.begin(), es.end());
#endif
        for (int dst: es) {
            if (res.find(dst) == res.end()) {
                res.insert(dst);
                queue.push_back(dst);
                ++count;
#ifdef MIN_NEIGHBOR
                {
                    ScopeLock l;
                    ++vc[dst];
                }
#endif
                if (count >= n_sample) {
                    break;
                }
            }
        }
    }

    return move(res);
}

unordered_set<int> uniform_sampler(int num_vertex, const Edges& edges, int n_sample) {
    unordered_set<int> res;
    while (res.size() < n_sample) {
        // original graph is 1-indexed
        res.insert(rand() % num_vertex + 1);
    }
    return move(res);
}

void dump(string file, const Edges& edges, unordered_set<int>& v) {
    // sort all sampled vertices
    vector<int> vertices;
    for (int node: v) {
        vertices.push_back(node);
    }
    sort(vertices.begin(), vertices.end());

    // build mapping from old to new
    unordered_map<int, int> map;
    for (int i = 0; i < vertices.size(); i++) {
        // vertex is 1-indexed
        map[vertices[i]] = i + 1;
    }
    
    // count all outgoing edges
    vector<int> count(vertices.size(), 0);
    Edges new_edges;
    for (auto& e: edges) {
        if (v.find(e.first) != v.end() && v.find(e.second) != v.end()) {
            // convert to new vertex index
            new_edges.emplace_back(map[e.first], map[e.second]);
            ++count[map[e.second] - 1];
        }
    }
    FILE* fp = fopen(file.c_str(), "w");
    fprintf(fp, "#\n#\n# Nodes: %lu Edges: %lu\n#\n", v.size(), new_edges.size());

    // print sampled node and # of outgoing edges
    for (int i = 0; i < vertices.size(); ++i) {
        fprintf(fp, "%d\t%d\n", vertices[i], count[i]);
    }

    // for sanity check
    fprintf(fp, "#\n");

    // print all edges (dst->src)
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
    string path = "samples85/";
    Sampler sampler = bfs_sampler;
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
    unordered_set<pair<int, int>, pairhash> edge_set;
    for (int i = 0; i < n_e; i++) {
        int src, dst;
        fscanf(fp, "%d\t%d\n", &src, &dst);
        // edges are stored in dst->src order
        edge_set.emplace(dst, src);
        // add reverse edge to make it undirected
        edge_set.emplace(src, dst);
    }
    fclose(fp);
    printf("good\n");
    Edges edges;
    for (auto& e: edge_set) {
        edges.push_back(e);
    }

    sort(edges.begin(), edges.end(), my_comparer);

#ifdef MIN_NEIGHBOR
    // bfs with minimum sampled neighbor, init mutex
    printf("bfs using min sampled neighbor\n");
    if (pthread_mutex_init(&mutex, NULL) != 0) {
        printf("\n mutex init failed\n");
        return -1;
    }
    for (int i = 1; i <= n_v; i++) {
        vc[i] = 0;
    }
#endif

    // init worker args
    int n_sample = int(n_v * PERCENT);
    vector<Arg> args;
    int each = NSAMPLE / NTHREAD;
    string prefix = path + "sample_";
    for (int i = 0; i < NTHREAD; i++) {
        vector<string> files;
        for (int j = 0; j < each; j++) {
            files.push_back(prefix + to_string(256+i * each + j) + ".txt");
        }
        args.emplace_back(edges, sampler, files, n_v, n_sample);
    }

    // create worker using multi-threading
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
