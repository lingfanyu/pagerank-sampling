#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <ctime>
#include <unordered_set>
#include <unordered_map>

using namespace std;

struct pairhash {
public:
    std::size_t operator()(const std::pair<int, int> &x) const{
        return std::hash<string>()(to_string(x.first) + "," + to_string(x.second));
    }
};

using Edges = vector<pair<int, int> >;
using VSet = unordered_set<int>;
using ESet = unordered_set<pair<int, int>, pairhash>;


unordered_set<int> bfs_sampler(int num_vertex, unordered_map<int, vector<int>>& adj, float percent) {
    int n_sample = int(num_vertex * percent);

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

unordered_set<int> uniform_sampler(int num_vertex, const Edges& edges, float percent) {
    int n_sample = int(num_vertex * percent);
    unordered_set<int> res;
    while (res.size() < n_sample) {
        // original graph is 1-indexed
        res.insert(rand() % num_vertex + 1);
    }
    return move(res);
}


void update_vertex(VSet& vs, ESet& es, const Edges& edges, const VSet&& v_sampled) {
    for (int v: v_sampled) {
        vs.insert(v);
    }
    for (auto& e: edges) {
        if (v_sampled.find(e.first) != v_sampled.end() && v_sampled.find(e.second) != v_sampled.end()) {
            es.insert(e);
        } 
    }
}

void edge_sampler(VSet& vs, ESet& es, const Edges& edges, float percent) {
    int n = edges.size();
    int n_sample = int(n * percent);
    unordered_set<int> res;
    while (res.size() < n_sample) {
        // original graph is 1-indexed
        res.insert(rand() % n);
    }
    for (int i: res) {
        auto& e = edges[i];
        es.insert(e);
        vs.insert(e.first);
    }
}

int main(int argc, const char** argv) {
    string method = "uniform";
    float percent = 0.01;
    if (argc == 3) {
        method = argv[1];
        percent = stof(argv[2]);
    }
    printf("%s sampling with p %f\n", method.c_str(), percent);
    srand(time(NULL));
    const char* filename = "web-Stanford.txt";

    // loading graph
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

    // build an adj matrix
    unordered_map<int, vector<int>> adj;
    for (auto& e: edges) {
        if (adj.find(e.first) == adj.end()) {
            adj[e.first] = vector<int>();
        }
        adj[e.first].push_back(e.second);
    }

    n_e = edges.size();
    VSet vs;
    ESet es;
    sprintf(buf, "%s_%.2f.txt", method.c_str(), percent);
    fp = fopen(buf, "w");
    fprintf(fp, "%d\t%d\n", n_v, n_e);
    fclose(fp);
    int n_vs = 0;
    int n_es = 0;
    int count = 0;
    while (n_vs < n_v * 0.99 || n_es < n_e * 0.99 || count < 1000) {
        if (method == "uniform") {
            update_vertex(vs, es, edges, uniform_sampler(n_v, edges, percent));
        } else if (method == "bfs") {
            update_vertex(vs, es, edges, bfs_sampler(n_v, adj, percent));
        } else if (method == "edge") {
            edge_sampler(vs, es, edges, percent);
        } else {
            printf("Unknown sampling method: %s\n", method.c_str());
            exit(-1);
        }
        n_vs = vs.size();
        n_es = es.size();
        fp = fopen(buf, "a");
        fprintf(fp, "%d\t%d\n", n_vs, n_es);
        fclose(fp);
        printf("sample %d\tv: %d / %d %.2f\te: %d / %d %.6f\n", count, n_vs, n_v, (float)n_vs / n_v, n_es, n_e, (float)n_es / n_e);
        ++count;
    }

    return 0;
}
