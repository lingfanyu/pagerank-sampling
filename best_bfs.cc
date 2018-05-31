#include <cstdio>
#include <cassert>
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

vector<int> bfs_sampler(int num_vertex, unordered_map<int, vector<int>>& adj, float percent, int start) {
    vector<int> visited(num_vertex + 1, 0);
    vector<int> queue;
    visited[start] = 1;
    queue.push_back(start);

    int n_sample = int(num_vertex * percent);
    int pos = 0;
    int count = 1;
    while (count < n_sample) {
        assert(pos < queue.size());
        int cur = queue[pos++];
        auto& es = adj[cur];
        random_shuffle(es.begin(), es.end());
        for (int dst: es) { 
            if (visited[dst] == 0) {
                visited[dst] = 1;
                queue.push_back(dst);
                ++count;
                if (count == n_sample) {
                    break;
                }
            }
        }
    }
    return move(queue);
}

int find_min_sampled_vertex(const vector<int>& vc) {
    int min_element = vc[1];
    for (int i = 2; i < vc.size(); i++) {
        if (min_element > vc[i]) {
            min_element = vc[i];
        }
    }
    vector<int> cand;
    for (int i = 1; i < vc.size(); i++) {
        if (vc[i] == min_element) {
            cand.push_back(i);
        }
    }
    return cand[rand() % cand.size()];
}

struct Args {
    string file;
    const Edges& edges;
    vector<int> vertices;
    Args(string f, Edges& e, vector<int>&& v) :
        file(f), edges(e), vertices(v) {}
};

void* dump(void* a) {
    Args* args = (Args*)a;

    // sort all sampled vertices
    auto& vertices = args->vertices;
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
    for (auto& e: (args->edges)) {
        if (map.find(e.first) != map.end() && map.find(e.second) != map.end()) {
            // convert to new vertex index
            new_edges.emplace_back(map[e.first], map[e.second]);
            ++count[map[e.second] - 1];
        }
    }
    FILE* fp = fopen(args->file.c_str(), "w");
    fprintf(fp, "#\n#\n# Nodes: %lu Edges: %lu\n#\n", vertices.size(), new_edges.size());

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
    free(args);
    return NULL;
}

int load_full_graph(Edges& edges) {
    const char* filename = "LCC.txt";
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

    for (auto& e: edge_set) {
        edges.push_back(e);
    }
    return n_v;
}

int main(int argc, const char** argv) {
    string prefix = "samples50/sample_";
    float percent = -0.01;
    int nsamples = 0;
    if (argc == 3) {
        percent = stof(argv[1]);
        nsamples = stoi(argv[2]);
    } else {
        printf("error usage\n");
        exit(-1);
    }
    printf("bfs sampling with p %f\n", percent);
    srand(time(NULL));

    // loading graph
    Edges edges;
    int n_v = load_full_graph(edges);

    // build an adj matrix
    unordered_map<int, vector<int>> adj;
    for (auto& e: edges) {
        if (adj.find(e.first) == adj.end()) {
            adj[e.first] = vector<int>();
        }
        adj[e.first].push_back(e.second);
    }

    // sampling
    vector<int> vc(n_v + 1, 0);
    vector<pthread_t> pids;
    for (int count = 0; count < nsamples; ++count) {
        int start = find_min_sampled_vertex(vc);
        auto v_sampled = bfs_sampler(n_v, adj, percent, start);
        for (int i: v_sampled) {
            ++vc[i];
        }
        Args* args = new Args(prefix + to_string(count) + ".txt", edges, move(v_sampled));
        pthread_t pid;
        pthread_create(&pid, NULL, dump, (void*)args);
        pids.push_back(pid);
        if (pids.size() >= 32) {
            for (auto& pid: pids) {
                pthread_join(pid, NULL);
            }
            pids.clear();
        }
    }

    FILE* fp = fopen("abcdefg.txt", "w");
    for (int i = 1; i <= n_v; i++) {
        fprintf(fp, "%d\n", vc[i]);
    }
    fclose(fp);
    for (auto& pid: pids) {
        pthread_join(pid, NULL);
    }

    return 0;
}
