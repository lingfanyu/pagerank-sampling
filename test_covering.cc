#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

using namespace std;

struct pairhash {
public:
    std::size_t operator()(const std::pair<int, int> &x) const{
        return std::hash<string>()(to_string(x.first) + "," + to_string(x.second));
    }
};

using Edges = vector<pair<int, int> >;
using VSet = unordered_set<int>;
using VCount = unordered_map<int, int>;
using ECount = unordered_map<pair<int, int>, int, pairhash>;

unordered_set<int> bfs_sampler(int num_vertex, unordered_map<int, vector<int>>& adj, float percent) {
    int n_sample = int(num_vertex * percent);

    unordered_set<int> res;

    // bfs
    vector<int> queue;

    // choose random start point
    // original graph is 1-indexed
    int start_node = rand() % num_vertex + 1;
    res.insert(start_node);
    queue.push_back(start_node);

    int pos = 0;
    int count = 1;
    while (pos < queue.size() && count < n_sample) {
        int cur = queue[pos++];
        auto& es = adj[cur];
        random_shuffle(es.begin(), es.end());
        for (int dst: es) { if (res.find(dst) == res.end()) {
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


void update_vertex(VCount& vs, ECount& es, const Edges& edges, const VSet&& v_sampled) {
    for (int v: v_sampled) {
        if (vs.find(v) == vs.end()) {
            vs[v] = 0;
        }
        ++vs[v];
    }
    /*
    for (auto& e: edges) {
        if (v_sampled.find(e.first) != v_sampled.end() && v_sampled.find(e.second) != v_sampled.end()) {
            if (es.find(e) == es.end()) {
                es[e] = 0;
            }
            ++es[e];
        } 
    }
    */
}

void edge_sampler(VCount& vs, ECount& es, const Edges& edges, float percent) {
    int n = edges.size();
    int n_sample = int(n * percent);
    unordered_set<int> res;
    while (res.size() < n_sample) {
        res.insert(rand() % n);
    }
    for (int i: res) {
        auto& e = edges[i];
        if (es.find(e) == es.end()) {
            es[e] = 0;
        }
        if (vs.find(e.first) == vs.end()) {
            vs[e.first] = 0;
        }
        ++vs[e.first];
        ++es[e];
    }
}

int main(int argc, const char** argv) {
    string method = "uniform";
    float percent = 0.01;
    int nsamples = 0;
    if (argc == 4) {
        method = argv[1];
        percent = stof(argv[2]);
        nsamples = stoi(argv[3]);
    } else {
        printf("error usage\n");
        exit(-1);
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
    VCount vs;
    ECount es;
    sprintf(buf, "tests/%s_%.2f_shuffle_stats.txt", method.c_str(), percent);
    //fp = fopen(buf, "w");
    //fprintf(fp, "%d\t%d\n", n_v, n_e);
    //fclose(fp);
    //int n_vs = 0;
    //int n_es = 0;
    int count = 0;
    int interval = nsamples / 10;
    while (count < nsamples) {
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
        //n_vs = vs.size();
        //n_es = es.size();
        //fp = fopen(buf, "a");
        //fprintf(fp, "%d\t%d\t%.2f\t%.6f\n", n_vs, n_es, (float)n_vs / n_v, (float)n_es / n_e);
        //fclose(fp);
        // printf("sample %d\tv: %d / %d %.2f\te: %d / %d %.6f\n", count, n_vs, n_v, (float)n_vs / n_v, n_es, n_e, (float)n_es / n_e);
        ++count;
        if (count % interval == 0) {
            printf("%d / %d\n", count, nsamples);
        }
    }
    
    fp = fopen(buf, "w");
    for (auto& it: vs) {
        fprintf(fp, "%d\t%d\n", it.first, it.second);
    }
    fprintf(fp, "#\n");
    /*
    for (auto& it: es) {
        fprintf(fp, "%d\t%d\t%d\n", it.first.first, it.first.second, it.second);
    }
    */
    fclose(fp);

    return 0;
}
