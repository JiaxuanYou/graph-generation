#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <ctime>
#include <iostream>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_map>
#include <algorithm>

using namespace std;

typedef long long int64;
typedef pair<int,int> PII;
typedef struct { int first, second, third; } TIII;

struct PAIR {
    int a, b;
    PAIR(int a0, int b0) { a=min(a0,b0); b=max(a0,b0); }
};
bool operator<(const PAIR &x, const PAIR &y) {
    if (x.a==y.a) return x.b<y.b;
    else return x.a<y.a;
}
bool operator==(const PAIR &x, const PAIR &y) {
    return x.a==y.a && x.b==y.b;
}
struct hash_PAIR {
    size_t operator()(const PAIR &x) const {
        return (x.a<<8) ^ (x.b<<0);
    }
};

struct TRIPLE {
    int a, b, c;
    TRIPLE(int a0, int b0, int c0) {
        a=a0; b=b0; c=c0;
        if (a>b) swap(a,b);
        if (b>c) swap(b,c);
        if (a>b) swap(a,b);
    }
};
bool operator<(const TRIPLE &x, const TRIPLE &y) {
    if (x.a==y.a) {
        if (x.b==y.b) return x.c<y.c;
        else return x.b<y.b;
    } else return x.a<y.a;
}
bool operator==(const TRIPLE &x, const TRIPLE &y) {
    return x.a==y.a && x.b==y.b && x.c==y.c;
}
struct hash_TRIPLE {
    size_t operator()(const TRIPLE &x) const {
        return (x.a<<16) ^ (x.b<<8) ^ (x.c<<0);
    }
};

unordered_map<PAIR, int, hash_PAIR> common2;
unordered_map<TRIPLE, int, hash_TRIPLE> common3;
unordered_map<PAIR, int, hash_PAIR>::iterator common2_it;
unordered_map<TRIPLE, int, hash_TRIPLE>::iterator common3_it;

#define common3_get(x) (((common3_it=common3.find(x))!=common3.end())?(common3_it->second):0)
#define common2_get(x) (((common2_it=common2.find(x))!=common2.end())?(common2_it->second):0)

int n,m; // n = number of nodes, m = number of edges
int *deg; // degrees of individual nodes
PAIR *edges; // list of edges

int **adj; // adj[x] - adjacency list of node x
PII **inc; // inc[x] - incidence list of node x: (y, edge id)
bool adjacent_list(int x, int y) { return binary_search(adj[x],adj[x]+deg[x],y); }
int *adj_matrix; // compressed adjacency matrix
const int adj_chunk = 8*sizeof(int);
bool adjacent_matrix(int x, int y) { return adj_matrix[(x*n+y)/adj_chunk]&(1<<((x*n+y)%adj_chunk)); }
bool (*adjacent)(int,int);
int getEdgeId(int x, int y) { return inc[x][lower_bound(adj[x],adj[x]+deg[x],y)-adj[x]].second; }

int64 **orbit; // orbit[x][o] - how many times does node x participate in orbit o
int64 **eorbit; // eorbit[x][o] - how many times does node x participate in edge orbit o

/** count graphlets on max 4 nodes */
void count4() {
    clock_t startTime, endTime;
    startTime = clock();
    clock_t startTime_all, endTime_all;
    startTime_all = startTime;
    int frac,frac_prev;

    // precompute triangles that span over edges
    printf("stage 1 - precomputing common nodes\n");
    int *tri = (int*)calloc(m,sizeof(int));
    frac_prev=-1;
    for (int i=0;i<m;i++) {
        frac = 100LL*i/m;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }
        int x=edges[i].a, y=edges[i].b;
        for (int xi=0,yi=0; xi<deg[x] && yi<deg[y]; ) {
            if (adj[x][xi]==adj[y][yi]) { tri[i]++; xi++; yi++; }
            else if (adj[x][xi]<adj[y][yi]) { xi++; }
            else { yi++; }
        }
    }
    endTime = clock();
    printf("%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
    startTime = endTime;

    // count full graphlets
    printf("stage 2 - counting full graphlets\n");
    int64 *C4 = (int64*)calloc(n,sizeof(int64));
    int *neigh = (int*)malloc(n*sizeof(int)), nn;
    frac_prev=-1;
    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }
        for (int nx=0;nx<deg[x];nx++) {
            int y=adj[x][nx];
            if (y >= x) break;
            nn=0;
            for (int ny=0;ny<deg[y];ny++) {
                int z=adj[y][ny];
                if (z >= y) break;
                if (adjacent(x,z)==0) continue;
                neigh[nn++]=z;
            }
            for (int i=0;i<nn;i++) {
                int z = neigh[i];
                for (int j=i+1;j<nn;j++) {
                    int zz = neigh[j];
                    if (adjacent(z,zz)) {
                        C4[x]++; C4[y]++; C4[z]++; C4[zz]++;
                    }
                }
            }
        }
    }
    endTime = clock();
    printf("%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
    startTime = endTime;

    // set up a system of equations relating orbits for every node
    printf("stage 3 - building systems of equations\n");
    int *common = (int*)calloc(n,sizeof(int));
    int *common_list = (int*)malloc(n*sizeof(int)), nc=0;
    frac_prev=-1;
    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }

        int64 f_12_14=0, f_10_13=0;
        int64 f_13_14=0, f_11_13=0;
        int64 f_7_11=0, f_5_8=0;
        int64 f_6_9=0, f_9_12=0, f_4_8=0, f_8_12=0;
        int64 f_14=C4[x];

        for (int i=0;i<nc;i++) common[common_list[i]]=0;
        nc=0;

        orbit[x][0]=deg[x];
        // x - middle node
        for (int nx1=0;nx1<deg[x];nx1++) {
            int y=inc[x][nx1].first, ey=inc[x][nx1].second;
            for (int ny=0;ny<deg[y];ny++) {
                int z=inc[y][ny].first, ez=inc[y][ny].second;
                if (adjacent(x,z)) { // triangle
                    if (z<y) {
                        f_12_14 += tri[ez]-1;
                        f_10_13 += (deg[y]-1-tri[ez])+(deg[z]-1-tri[ez]);
                    }
                } else {
                    if (common[z]==0) common_list[nc++]=z;
                    common[z]++;
                }
            }
            for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                int z=inc[x][nx2].first, ez=inc[x][nx2].second;
                if (adjacent(y,z)) { // triangle
                    orbit[x][3]++;
                    f_13_14 += (tri[ey]-1)+(tri[ez]-1);
                    f_11_13 += (deg[x]-1-tri[ey])+(deg[x]-1-tri[ez]);
                } else { // path
                    orbit[x][2]++;
                    f_7_11 += (deg[x]-1-tri[ey]-1)+(deg[x]-1-tri[ez]-1);
                    f_5_8 += (deg[y]-1-tri[ey])+(deg[z]-1-tri[ez]);
                }
            }
        }
        // x - side node
        for (int nx1=0;nx1<deg[x];nx1++) {
            int y=inc[x][nx1].first, ey=inc[x][nx1].second;
            for (int ny=0;ny<deg[y];ny++) {
                int z=inc[y][ny].first, ez=inc[y][ny].second;
                if (x==z) continue;
                if (!adjacent(x,z)) { // path
                    orbit[x][1]++;
                    f_6_9 += (deg[y]-1-tri[ey]-1);
                    f_9_12 += tri[ez];
                    f_4_8 += (deg[z]-1-tri[ez]);
                    f_8_12 += (common[z]-1);
                }
            }
        }

        // solve system of equations
        orbit[x][14]=(f_14);
        orbit[x][13]=(f_13_14-6*f_14)/2;
        orbit[x][12]=(f_12_14-3*f_14);
        orbit[x][11]=(f_11_13-f_13_14+6*f_14)/2;
        orbit[x][10]=(f_10_13-f_13_14+6*f_14);
        orbit[x][9]=(f_9_12-2*f_12_14+6*f_14)/2;
        orbit[x][8]=(f_8_12-2*f_12_14+6*f_14)/2;
        orbit[x][7]=(f_13_14+f_7_11-f_11_13-6*f_14)/6;
        orbit[x][6]=(2*f_12_14+f_6_9-f_9_12-6*f_14)/2;
        orbit[x][5]=(2*f_12_14+f_5_8-f_8_12-6*f_14);
        orbit[x][4]=(2*f_12_14+f_4_8-f_8_12-6*f_14);
    }

    endTime = clock();
    printf("%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);

    endTime_all = endTime;
    printf("total: %.2f\n", (double)(endTime_all-startTime_all)/CLOCKS_PER_SEC);
}


/** count edge orbits of graphlets on max 4 nodes */
void ecount4() {
    clock_t startTime, endTime;
    startTime = clock();
    clock_t startTime_all, endTime_all;
    startTime_all = startTime;
    int frac,frac_prev;

    // precompute triangles that span over edges
    printf("stage 1 - precomputing common nodes\n");
    int *tri = (int*)calloc(m,sizeof(int));
    frac_prev=-1;
    for (int i=0;i<m;i++) {
        frac = 100LL*i/m;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }
        int x=edges[i].a, y=edges[i].b;
        for (int xi=0,yi=0; xi<deg[x] && yi<deg[y]; ) {
            if (adj[x][xi]==adj[y][yi]) { tri[i]++; xi++; yi++; }
            else if (adj[x][xi]<adj[y][yi]) { xi++; }
            else { yi++; }
        }
    }
    endTime = clock();
    printf("%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
    startTime = endTime;

    // count full graphlets
    printf("stage 2 - counting full graphlets\n");
    int64 *C4 = (int64*)calloc(m,sizeof(int64));
    int *neighx = (int*)malloc(n*sizeof(int)); // lookup table - edges to neighbors of x
    memset(neighx,-1,n*sizeof(int));
    int *neigh = (int*)malloc(n*sizeof(int)), nn; // lookup table - common neighbors of x and y
    PII *neigh_edges = (PII*)malloc(n*sizeof(PII)); // list of common neighbors of x and y
    frac_prev=-1;
    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }

        for (int nx=0;nx<deg[x];nx++) {
            int y=inc[x][nx].first, xy=inc[x][nx].second;
            neighx[y]=xy;
        }
        for (int nx=0;nx<deg[x];nx++) {
            int y=inc[x][nx].first, xy=inc[x][nx].second;
            if (y >= x) break;
            nn=0;
            for (int ny=0;ny<deg[y];ny++) {
                int z=inc[y][ny].first, yz=inc[y][ny].second;
                if (z >= y) break;
                if (neighx[z]==-1) continue;
                int xz=neighx[z];
                neigh[nn]=z;
                neigh_edges[nn]={xz, yz};
                nn++;
            }
            for (int i=0;i<nn;i++) {
                int z = neigh[i], xz = neigh_edges[i].first, yz = neigh_edges[i].second;
                for (int j=i+1;j<nn;j++) {
                    int w = neigh[j], xw = neigh_edges[j].first, yw = neigh_edges[j].second;
                    if (adjacent(z,w)) {
                        C4[xy]++;
                        C4[xz]++; C4[yz]++;
                        C4[xw]++; C4[yw]++;
                        // another iteration to count this last(smallest) edge instead of calling getEdgeId
                        //int zw=getEdgeId(z,w); C4[zw]++;
                    }
                }
            }
        }
        for (int nx=0;nx<deg[x];nx++) {
            int y=inc[x][nx].first, xy=inc[x][nx].second;
            neighx[y]=-1;
        }
    }
    endTime = clock();
    printf("%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
    startTime = endTime;

    // count full graphlets for the smallest edge
    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }
        for (int nx=deg[x]-1;nx>=0;nx--) {
            int y=inc[x][nx].first, xy=inc[x][nx].second;
            if (y <= x) break;
            nn=0;
            for (int ny=deg[y]-1;ny>=0;ny--) {
                int z=adj[y][ny];
                if (z <= y) break;
                if (adjacent(x,z)==0) continue;
                neigh[nn++]=z;
            }
            for (int i=0;i<nn;i++) {
                int z = neigh[i];
                for (int j=i+1;j<nn;j++) {
                    int zz = neigh[j];
                    if (adjacent(z,zz)) {
                        C4[xy]++;
                    }
                }
            }
        }
    }
    endTime = clock();
    printf("%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
    startTime = endTime;

    // set up a system of equations relating orbits for every node
    printf("stage 3 - building systems of equations\n");
    int *common = (int*)calloc(n,sizeof(int));
    int *common_list = (int*)malloc(n*sizeof(int)), nc=0;
    frac_prev=-1;

    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }

        // common nodes of x and some other node
        for (int i=0;i<nc;i++) common[common_list[i]]=0;
        nc=0;
        for (int nx=0;nx<deg[x];nx++) {
            int y=adj[x][nx];
            for (int ny=0;ny<deg[y];ny++) {
                int z=adj[y][ny];
                if (z==x) continue;
                if (common[z]==0) common_list[nc++]=z;
                common[z]++;
            }
        }

        for (int nx=0;nx<deg[x];nx++) {
            int y=inc[x][nx].first, xy=inc[x][nx].second;
            int e=xy;
            for (int n1=0;n1<deg[x];n1++) {
                int z=inc[x][n1].first, xz=inc[x][n1].second;
                if (z==y) continue;
                if (adjacent(y,z)) { // triangle
                    if (x<y) {
                        eorbit[e][1]++;
                        eorbit[e][10] += tri[xy]-1;
                        eorbit[e][7] += deg[z]-2;
                    }
                    eorbit[e][9] += tri[xz]-1;
                    eorbit[e][8] += deg[x]-2;
                }
            }
            for (int n1=0;n1<deg[y];n1++) {
                int z=inc[y][n1].first, yz=inc[y][n1].second;
                if (z==x) continue;
                if (!adjacent(x,z)) { // path x-y-z
                    eorbit[e][0]++;
                    eorbit[e][6] += tri[yz];
                    eorbit[e][5] += common[z]-1;
                    eorbit[e][4] += deg[y]-2;
                    eorbit[e][3] += deg[x]-1;
                    eorbit[e][2] += deg[z]-1;
                }
            }
        }
    }
    // solve system of equations
    for (int e=0;e<m;e++) {
        eorbit[e][11]=C4[e];
        eorbit[e][10]=(eorbit[e][10]-2*eorbit[e][11])/2;
        eorbit[e][9]=(eorbit[e][9]-4*eorbit[e][11]);
        eorbit[e][8]=(eorbit[e][8]-eorbit[e][9]-4*eorbit[e][10]-4*eorbit[e][11]);
        eorbit[e][7]=(eorbit[e][7]-eorbit[e][9]-2*eorbit[e][11]);
        eorbit[e][6]=(eorbit[e][6]-eorbit[e][9])/2;
        eorbit[e][5]=(eorbit[e][5]-eorbit[e][9])/2;
        eorbit[e][4]=(eorbit[e][4]-2*eorbit[e][6]-eorbit[e][8]-eorbit[e][9])/2;
        eorbit[e][3]=(eorbit[e][3]-2*eorbit[e][5]-eorbit[e][8]-eorbit[e][9])/2;
        eorbit[e][2]=(eorbit[e][2]-2*eorbit[e][5]-2*eorbit[e][6]-eorbit[e][9]);
    }

    endTime = clock();
    printf("%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);

    endTime_all = endTime;
    printf("total: %.2f\n", (double)(endTime_all-startTime_all)/CLOCKS_PER_SEC);
}


/** count graphlets on max 5 nodes */
void count5() {
    clock_t startTime, endTime;
    startTime = clock();
    clock_t startTime_all, endTime_all;
    startTime_all = startTime;
    int frac,frac_prev;

    // precompute common nodes
    printf("stage 1 - precomputing common nodes\n");
    frac_prev=-1;
    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }
        for (int n1=0;n1<deg[x];n1++) {
            int a=adj[x][n1];
            for (int n2=n1+1;n2<deg[x];n2++) {
                int b=adj[x][n2];
                PAIR ab=PAIR(a,b);
                common2[ab]++;
                for (int n3=n2+1;n3<deg[x];n3++) {
                    int c=adj[x][n3];
                    int st = adjacent(a,b)+adjacent(a,c)+adjacent(b,c);
                    if (st<2) continue;
                    TRIPLE abc=TRIPLE(a,b,c);
                    common3[abc]++;
                }
            }
        }
    }
    // precompute triangles that span over edges
    int *tri = (int*)calloc(m,sizeof(int));
    for (int i=0;i<m;i++) {
        int x=edges[i].a, y=edges[i].b;
        for (int xi=0,yi=0; xi<deg[x] && yi<deg[y]; ) {
            if (adj[x][xi]==adj[y][yi]) { tri[i]++; xi++; yi++; }
            else if (adj[x][xi]<adj[y][yi]) { xi++; }
            else { yi++; }
        }
    }
    endTime = clock();
    printf("%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
    startTime = endTime;

    // count full graphlets
    printf("stage 2 - counting full graphlets\n");
    int64 *C5 = (int64*)calloc(n,sizeof(int64));
    int *neigh = (int*)malloc(n*sizeof(int)), nn;
    int *neigh2 = (int*)malloc(n*sizeof(int)), nn2;
    frac_prev=-1;
    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }
        for (int nx=0;nx<deg[x];nx++) {
            int y=adj[x][nx];
            if (y >= x) break;
            nn=0;
            for (int ny=0;ny<deg[y];ny++) {
                int z=adj[y][ny];
                if (z >= y) break;
                if (adjacent(x,z)) {
                    neigh[nn++]=z;
                }
            }
            for (int i=0;i<nn;i++) {
                int z = neigh[i];
                nn2=0;
                for (int j=i+1;j<nn;j++) {
                    int zz = neigh[j];
                    if (adjacent(z,zz)) {
                        neigh2[nn2++]=zz;
                    }
                }
                for (int i2=0;i2<nn2;i2++) {
                    int zz = neigh2[i2];
                    for (int j2=i2+1;j2<nn2;j2++) {
                        int zzz = neigh2[j2];
                        if (adjacent(zz,zzz)) {
                            C5[x]++; C5[y]++; C5[z]++; C5[zz]++; C5[zzz]++;
                        }
                    }
                }
            }
        }
    }
    endTime = clock();
    printf("%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
    startTime = endTime;

    int *common_x = (int*)calloc(n,sizeof(int));
    int *common_x_list = (int*)malloc(n*sizeof(int)), ncx=0;
    int *common_a = (int*)calloc(n,sizeof(int));
    int *common_a_list = (int*)malloc(n*sizeof(int)), nca=0;

    // set up a system of equations relating orbit counts
    printf("stage 3 - building systems of equations\n");
    frac_prev=-1;
    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }

        for (int i=0;i<ncx;i++) common_x[common_x_list[i]]=0;
        ncx=0;

        // smaller graphlets
        orbit[x][0] = deg[x];
        for (int nx1=0;nx1<deg[x];nx1++) {
            int a=adj[x][nx1];
            for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                int b=adj[x][nx2];
                if (adjacent(a,b)) orbit[x][3]++;
                else orbit[x][2]++;
            }
            for (int na=0;na<deg[a];na++) {
                int b=adj[a][na];
                if (b!=x && !adjacent(x,b)) {
                    orbit[x][1]++;
                    if (common_x[b]==0) common_x_list[ncx++]=b;
                    common_x[b]++;
                }
            }
        }

        int64 f_71=0, f_70=0, f_67=0, f_66=0, f_58=0, f_57=0; // 14
        int64 f_69=0, f_68=0, f_64=0, f_61=0, f_60=0, f_55=0, f_48=0, f_42=0, f_41=0; // 13
        int64 f_65=0, f_63=0, f_59=0, f_54=0, f_47=0, f_46=0, f_40=0; // 12
        int64 f_62=0, f_53=0, f_51=0, f_50=0, f_49=0, f_38=0, f_37=0, f_36=0; // 8
        int64 f_44=0, f_33=0, f_30=0, f_26=0; // 11
        int64 f_52=0, f_43=0, f_32=0, f_29=0, f_25=0; // 10
        int64 f_56=0, f_45=0, f_39=0, f_31=0, f_28=0, f_24=0; // 9
        int64 f_35=0, f_34=0, f_27=0, f_18=0, f_16=0, f_15=0; // 4
        int64 f_17=0; // 5
        int64 f_22=0, f_20=0, f_19=0; // 6
        int64 f_23=0, f_21=0; // 7

        for (int nx1=0;nx1<deg[x];nx1++) {
            int a=inc[x][nx1].first, xa=inc[x][nx1].second;

            for (int i=0;i<nca;i++) common_a[common_a_list[i]]=0;
            nca=0;
            for (int na=0;na<deg[a];na++) {
                int b=adj[a][na];
                for (int nb=0;nb<deg[b];nb++) {
                    int c=adj[b][nb];
                    if (c==a || adjacent(a,c)) continue;
                    if (common_a[c]==0) common_a_list[nca++]=c;
                    common_a[c]++;
                }
            }

            // x = orbit-14 (tetrahedron)
            for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                int b=inc[x][nx2].first, xb=inc[x][nx2].second;
                if (!adjacent(a,b)) continue;
                for (int nx3=nx2+1;nx3<deg[x];nx3++) {
                    int c=inc[x][nx3].first, xc=inc[x][nx3].second;
                    if (!adjacent(a,c) || !adjacent(b,c)) continue;
                    orbit[x][14]++;
                    f_70 += common3_get(TRIPLE(a,b,c))-1;
                    f_71 += (tri[xa]>2 && tri[xb]>2)?(common3_get(TRIPLE(x,a,b))-1):0;
                    f_71 += (tri[xa]>2 && tri[xc]>2)?(common3_get(TRIPLE(x,a,c))-1):0;
                    f_71 += (tri[xb]>2 && tri[xc]>2)?(common3_get(TRIPLE(x,b,c))-1):0;
                    f_67 += tri[xa]-2+tri[xb]-2+tri[xc]-2;
                    f_66 += common2_get(PAIR(a,b))-2;
                    f_66 += common2_get(PAIR(a,c))-2;
                    f_66 += common2_get(PAIR(b,c))-2;
                    f_58 += deg[x]-3;
                    f_57 += deg[a]-3+deg[b]-3+deg[c]-3;
                }
            }

            // x = orbit-13 (diamond)
            for (int nx2=0;nx2<deg[x];nx2++) {
                int b=inc[x][nx2].first, xb=inc[x][nx2].second;
                if (!adjacent(a,b)) continue;
                for (int nx3=nx2+1;nx3<deg[x];nx3++) {
                    int c=inc[x][nx3].first, xc=inc[x][nx3].second;
                    if (!adjacent(a,c) || adjacent(b,c)) continue;
                    orbit[x][13]++;
                    f_69 += (tri[xb]>1 && tri[xc]>1)?(common3_get(TRIPLE(x,b,c))-1):0;
                    f_68 += common3_get(TRIPLE(a,b,c))-1;
                    f_64 += common2_get(PAIR(b,c))-2;
                    f_61 += tri[xb]-1+tri[xc]-1;
                    f_60 += common2_get(PAIR(a,b))-1;
                    f_60 += common2_get(PAIR(a,c))-1;
                    f_55 += tri[xa]-2;
                    f_48 += deg[b]-2+deg[c]-2;
                    f_42 += deg[x]-3;
                    f_41 += deg[a]-3;
                }
            }

            // x = orbit-12 (diamond)
            for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                int b=inc[x][nx2].first, xb=inc[x][nx2].second;
                if (!adjacent(a,b)) continue;
                for (int na=0;na<deg[a];na++) {
                    int c=inc[a][na].first, ac=inc[a][na].second;
                    if (c==x || adjacent(x,c) || !adjacent(b,c)) continue;
                    orbit[x][12]++;
                    f_65 += (tri[ac]>1)?common3_get(TRIPLE(a,b,c)):0;
                    f_63 += common_x[c]-2;
                    f_59 += tri[ac]-1+common2_get(PAIR(b,c))-1;
                    f_54 += common2_get(PAIR(a,b))-2;
                    f_47 += deg[x]-2;
                    f_46 += deg[c]-2;
                    f_40 += deg[a]-3+deg[b]-3;
                }
            }

            // x = orbit-8 (cycle)
            for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                int b=inc[x][nx2].first, xb=inc[x][nx2].second;
                if (adjacent(a,b)) continue;
                for (int na=0;na<deg[a];na++) {
                    int c=inc[a][na].first, ac=inc[a][na].second;
                    if (c==x || adjacent(x,c) || !adjacent(b,c)) continue;
                    orbit[x][8]++;
                    f_62 += (tri[ac]>0)?common3_get(TRIPLE(a,b,c)):0;
                    f_53 += tri[xa]+tri[xb];
                    f_51 += tri[ac]+common2_get(PAIR(c,b));
                    f_50 += common_x[c]-2;
                    f_49 += common_a[b]-2;
                    f_38 += deg[x]-2;
                    f_37 += deg[a]-2+deg[b]-2;
                    f_36 += deg[c]-2;
                }
            }

            // x = orbit-11 (paw)
            for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                int b=inc[x][nx2].first, xb=inc[x][nx2].second;
                if (!adjacent(a,b)) continue;
                for (int nx3=0;nx3<deg[x];nx3++) {
                    int c=inc[x][nx3].first, xc=inc[x][nx3].second;
                    if (c==a || c==b || adjacent(a,c) || adjacent(b,c)) continue;
                    orbit[x][11]++;
                    f_44 += tri[xc];
                    f_33 += deg[x]-3;
                    f_30 += deg[c]-1;
                    f_26 += deg[a]-2+deg[b]-2;
                }
            }

            // x = orbit-10 (paw)
            for (int nx2=0;nx2<deg[x];nx2++) {
                int b=inc[x][nx2].first, xb=inc[x][nx2].second;
                if (!adjacent(a,b)) continue;
                for (int nb=0;nb<deg[b];nb++) {
                    int c=inc[b][nb].first, bc=inc[b][nb].second;
                    if (c==x || c==a || adjacent(a,c) || adjacent(x,c)) continue;
                    orbit[x][10]++;
                    f_52 += common_a[c]-1;
                    f_43 += tri[bc];
                    f_32 += deg[b]-3;
                    f_29 += deg[c]-1;
                    f_25 += deg[a]-2;
                }
            }

            // x = orbit-9 (paw)
            for (int na1=0;na1<deg[a];na1++) {
                int b=inc[a][na1].first, ab=inc[a][na1].second;
                if (b==x || adjacent(x,b)) continue;
                for (int na2=na1+1;na2<deg[a];na2++) {
                    int c=inc[a][na2].first, ac=inc[a][na2].second;
                    if (c==x || !adjacent(b,c) || adjacent(x,c)) continue;
                    orbit[x][9]++;
                    f_56 += (tri[ab]>1 && tri[ac]>1)?common3_get(TRIPLE(a,b,c)):0;
                    f_45 += common2_get(PAIR(b,c))-1;
                    f_39 += tri[ab]-1+tri[ac]-1;
                    f_31 += deg[a]-3;
                    f_28 += deg[x]-1;
                    f_24 += deg[b]-2+deg[c]-2;
                }
            }

            // x = orbit-4 (path)
            for (int na=0;na<deg[a];na++) {
                int b=inc[a][na].first, ab=inc[a][na].second;
                if (b==x || adjacent(x,b)) continue;
                for (int nb=0;nb<deg[b];nb++) {
                    int c=inc[b][nb].first, bc=inc[b][nb].second;
                    if (c==a || adjacent(a,c) || adjacent(x,c)) continue;
                    orbit[x][4]++;
                    f_35 += common_a[c]-1;
                    f_34 += common_x[c];
                    f_27 += tri[bc];
                    f_18 += deg[b]-2;
                    f_16 += deg[x]-1;
                    f_15 += deg[c]-1;
                }
            }

            // x = orbit-5 (path)
            for (int nx2=0;nx2<deg[x];nx2++) {
                int b=inc[x][nx2].first, xb=inc[x][nx2].second;
                if (b==a || adjacent(a,b)) continue;
                for (int nb=0;nb<deg[b];nb++) {
                    int c=inc[b][nb].first, bc=inc[b][nb].second;
                    if (c==x || adjacent(a,c) || adjacent(x,c)) continue;
                    orbit[x][5]++;
                    f_17 += deg[a]-1;
                }
            }

            // x = orbit-6 (claw)
            for (int na1=0;na1<deg[a];na1++) {
                int b=inc[a][na1].first, ab=inc[a][na1].second;
                if (b==x || adjacent(x,b)) continue;
                for (int na2=na1+1;na2<deg[a];na2++) {
                    int c=inc[a][na2].first, ac=inc[a][na2].second;
                    if (c==x || adjacent(x,c) || adjacent(b,c)) continue;
                    orbit[x][6]++;
                    f_22 += deg[a]-3;
                    f_20 += deg[x]-1;
                    f_19 += deg[b]-1+deg[c]-1;
                }
            }

            // x = orbit-7 (claw)
            for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                int b=inc[x][nx2].first, xb=inc[x][nx2].second;
                if (adjacent(a,b)) continue;
                for (int nx3=nx2+1;nx3<deg[x];nx3++) {
                    int c=inc[x][nx3].first, xc=inc[x][nx3].second;
                    if (adjacent(a,c) || adjacent(b,c)) continue;
                    orbit[x][7]++;
                    f_23 += deg[x]-3;
                    f_21 += deg[a]-1+deg[b]-1+deg[c]-1;
                }
            }
        }

        // solve equations
        orbit[x][72] = C5[x];
        orbit[x][71] = (f_71-12*orbit[x][72])/2;
        orbit[x][70] = (f_70-4*orbit[x][72]);
        orbit[x][69] = (f_69-2*orbit[x][71])/4;
        orbit[x][68] = (f_68-2*orbit[x][71]);
        orbit[x][67] = (f_67-12*orbit[x][72]-4*orbit[x][71]);
        orbit[x][66] = (f_66-12*orbit[x][72]-2*orbit[x][71]-3*orbit[x][70]);
        orbit[x][65] = (f_65-3*orbit[x][70])/2;
        orbit[x][64] = (f_64-2*orbit[x][71]-4*orbit[x][69]-1*orbit[x][68]);
        orbit[x][63] = (f_63-3*orbit[x][70]-2*orbit[x][68]);
        orbit[x][62] = (f_62-1*orbit[x][68])/2;
        orbit[x][61] = (f_61-4*orbit[x][71]-8*orbit[x][69]-2*orbit[x][67])/2;
        orbit[x][60] = (f_60-4*orbit[x][71]-2*orbit[x][68]-2*orbit[x][67]);
        orbit[x][59] = (f_59-6*orbit[x][70]-2*orbit[x][68]-4*orbit[x][65]);
        orbit[x][58] = (f_58-4*orbit[x][72]-2*orbit[x][71]-1*orbit[x][67]);
        orbit[x][57] = (f_57-12*orbit[x][72]-4*orbit[x][71]-3*orbit[x][70]-1*orbit[x][67]-2*orbit[x][66]);
        orbit[x][56] = (f_56-2*orbit[x][65])/3;
        orbit[x][55] = (f_55-2*orbit[x][71]-2*orbit[x][67])/3;
        orbit[x][54] = (f_54-3*orbit[x][70]-1*orbit[x][66]-2*orbit[x][65])/2;
        orbit[x][53] = (f_53-2*orbit[x][68]-2*orbit[x][64]-2*orbit[x][63]);
        orbit[x][52] = (f_52-2*orbit[x][66]-2*orbit[x][64]-1*orbit[x][59])/2;
        orbit[x][51] = (f_51-2*orbit[x][68]-2*orbit[x][63]-4*orbit[x][62]);
        orbit[x][50] = (f_50-1*orbit[x][68]-2*orbit[x][63])/3;
        orbit[x][49] = (f_49-1*orbit[x][68]-1*orbit[x][64]-2*orbit[x][62])/2;
        orbit[x][48] = (f_48-4*orbit[x][71]-8*orbit[x][69]-2*orbit[x][68]-2*orbit[x][67]-2*orbit[x][64]-2*orbit[x][61]-1*orbit[x][60]);
        orbit[x][47] = (f_47-3*orbit[x][70]-2*orbit[x][68]-1*orbit[x][66]-1*orbit[x][63]-1*orbit[x][60]);
        orbit[x][46] = (f_46-3*orbit[x][70]-2*orbit[x][68]-2*orbit[x][65]-1*orbit[x][63]-1*orbit[x][59]);
        orbit[x][45] = (f_45-2*orbit[x][65]-2*orbit[x][62]-3*orbit[x][56]);
        orbit[x][44] = (f_44-1*orbit[x][67]-2*orbit[x][61])/4;
        orbit[x][43] = (f_43-2*orbit[x][66]-1*orbit[x][60]-1*orbit[x][59])/2;
        orbit[x][42] = (f_42-2*orbit[x][71]-4*orbit[x][69]-2*orbit[x][67]-2*orbit[x][61]-3*orbit[x][55]);
        orbit[x][41] = (f_41-2*orbit[x][71]-1*orbit[x][68]-2*orbit[x][67]-1*orbit[x][60]-3*orbit[x][55]);
        orbit[x][40] = (f_40-6*orbit[x][70]-2*orbit[x][68]-2*orbit[x][66]-4*orbit[x][65]-1*orbit[x][60]-1*orbit[x][59]-4*orbit[x][54]);
        orbit[x][39] = (f_39-4*orbit[x][65]-1*orbit[x][59]-6*orbit[x][56])/2;
        orbit[x][38] = (f_38-1*orbit[x][68]-1*orbit[x][64]-2*orbit[x][63]-1*orbit[x][53]-3*orbit[x][50]);
        orbit[x][37] = (f_37-2*orbit[x][68]-2*orbit[x][64]-2*orbit[x][63]-4*orbit[x][62]-1*orbit[x][53]-1*orbit[x][51]-4*orbit[x][49]);
        orbit[x][36] = (f_36-1*orbit[x][68]-2*orbit[x][63]-2*orbit[x][62]-1*orbit[x][51]-3*orbit[x][50]);
        orbit[x][35] = (f_35-1*orbit[x][59]-2*orbit[x][52]-2*orbit[x][45])/2;
        orbit[x][34] = (f_34-1*orbit[x][59]-2*orbit[x][52]-1*orbit[x][51])/2;
        orbit[x][33] = (f_33-1*orbit[x][67]-2*orbit[x][61]-3*orbit[x][58]-4*orbit[x][44]-2*orbit[x][42])/2;
        orbit[x][32] = (f_32-2*orbit[x][66]-1*orbit[x][60]-1*orbit[x][59]-2*orbit[x][57]-2*orbit[x][43]-2*orbit[x][41]-1*orbit[x][40])/2;
        orbit[x][31] = (f_31-2*orbit[x][65]-1*orbit[x][59]-3*orbit[x][56]-1*orbit[x][43]-2*orbit[x][39]);
        orbit[x][30] = (f_30-1*orbit[x][67]-1*orbit[x][63]-2*orbit[x][61]-1*orbit[x][53]-4*orbit[x][44]);
        orbit[x][29] = (f_29-2*orbit[x][66]-2*orbit[x][64]-1*orbit[x][60]-1*orbit[x][59]-1*orbit[x][53]-2*orbit[x][52]-2*orbit[x][43]);
        orbit[x][28] = (f_28-2*orbit[x][65]-2*orbit[x][62]-1*orbit[x][59]-1*orbit[x][51]-1*orbit[x][43]);
        orbit[x][27] = (f_27-1*orbit[x][59]-1*orbit[x][51]-2*orbit[x][45])/2;
        orbit[x][26] = (f_26-2*orbit[x][67]-2*orbit[x][63]-2*orbit[x][61]-6*orbit[x][58]-1*orbit[x][53]-2*orbit[x][47]-2*orbit[x][42]);
        orbit[x][25] = (f_25-2*orbit[x][66]-2*orbit[x][64]-1*orbit[x][59]-2*orbit[x][57]-2*orbit[x][52]-1*orbit[x][48]-1*orbit[x][40])/2;
        orbit[x][24] = (f_24-4*orbit[x][65]-4*orbit[x][62]-1*orbit[x][59]-6*orbit[x][56]-1*orbit[x][51]-2*orbit[x][45]-2*orbit[x][39]);
        orbit[x][23] = (f_23-1*orbit[x][55]-1*orbit[x][42]-2*orbit[x][33])/4;
        orbit[x][22] = (f_22-2*orbit[x][54]-1*orbit[x][40]-1*orbit[x][39]-1*orbit[x][32]-2*orbit[x][31])/3;
        orbit[x][21] = (f_21-3*orbit[x][55]-3*orbit[x][50]-2*orbit[x][42]-2*orbit[x][38]-2*orbit[x][33]);
        orbit[x][20] = (f_20-2*orbit[x][54]-2*orbit[x][49]-1*orbit[x][40]-1*orbit[x][37]-1*orbit[x][32]);
        orbit[x][19] = (f_19-4*orbit[x][54]-4*orbit[x][49]-1*orbit[x][40]-2*orbit[x][39]-1*orbit[x][37]-2*orbit[x][35]-2*orbit[x][31]);
        orbit[x][18] = (f_18-1*orbit[x][59]-1*orbit[x][51]-2*orbit[x][46]-2*orbit[x][45]-2*orbit[x][36]-2*orbit[x][27]-1*orbit[x][24])/2;
        orbit[x][17] = (f_17-1*orbit[x][60]-1*orbit[x][53]-1*orbit[x][51]-1*orbit[x][48]-1*orbit[x][37]-2*orbit[x][34]-2*orbit[x][30])/2;
        orbit[x][16] = (f_16-1*orbit[x][59]-2*orbit[x][52]-1*orbit[x][51]-2*orbit[x][46]-2*orbit[x][36]-2*orbit[x][34]-1*orbit[x][29]);
        orbit[x][15] = (f_15-1*orbit[x][59]-2*orbit[x][52]-1*orbit[x][51]-2*orbit[x][45]-2*orbit[x][35]-2*orbit[x][34]-2*orbit[x][27]);
    }
    endTime = clock();
    printf("%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);

    endTime_all = endTime;
    printf("total: %.2f sec\n", (double)(endTime_all-startTime_all)/CLOCKS_PER_SEC);
}


/** count edge orbits of graphlets on max 5 nodes */
void ecount5() {
    clock_t startTime, endTime;
    startTime = clock();
    clock_t startTime_all, endTime_all;
    startTime_all = startTime;
    int frac,frac_prev;

    // precompute common nodes
    printf("stage 1 - precomputing common nodes\n");
    frac_prev=-1;
    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }
        for (int n1=0;n1<deg[x];n1++) {
            int a=adj[x][n1];
            for (int n2=n1+1;n2<deg[x];n2++) {
                int b=adj[x][n2];
                PAIR ab=PAIR(a,b);
                common2[ab]++;
                for (int n3=n2+1;n3<deg[x];n3++) {
                    int c=adj[x][n3];
                    int st = adjacent(a,b)+adjacent(a,c)+adjacent(b,c);
                    if (st<2) continue;
                    TRIPLE abc=TRIPLE(a,b,c);
                    common3[abc]++;
                }
            }
        }
    }
    // precompute triangles that span over edges
    int *tri = (int*)calloc(m,sizeof(int));
    for (int i=0;i<m;i++) {
        int x=edges[i].a, y=edges[i].b;
        for (int xi=0,yi=0; xi<deg[x] && yi<deg[y]; ) {
            if (adj[x][xi]==adj[y][yi]) { tri[i]++; xi++; yi++; }
            else if (adj[x][xi]<adj[y][yi]) { xi++; }
            else { yi++; }
        }
    }
    endTime = clock();
    printf("%.2f sec\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
    startTime = endTime;

    // count full graphlets
    printf("stage 2 - counting full graphlets\n");
    int64 *C5 = (int64*)calloc(m,sizeof(int64));
    int *neighx = (int*)malloc(n*sizeof(int)); // lookup table - edges to neighbors of x
    memset(neighx,-1,n*sizeof(int));
    int *neigh = (int*)malloc(n*sizeof(int)), nn; // lookup table - common neighbors of x and y
    PII *neigh_edges = (PII*)malloc(n*sizeof(PII)); // list of common neighbors of x and y
    int *neigh2 = (int*)malloc(n*sizeof(int)), nn2;
    TIII *neigh2_edges = (TIII*)malloc(n*sizeof(TIII));
    frac_prev=-1;
    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }

        for (int nx=0;nx<deg[x];nx++) {
            int y=inc[x][nx].first, xy=inc[x][nx].second;
            neighx[y]=xy;
        }
        for (int nx=0;nx<deg[x];nx++) {
            int y=inc[x][nx].first, xy=inc[x][nx].second;
            if (y >= x) break;
            nn=0;
            for (int ny=0;ny<deg[y];ny++) {
                int z=inc[y][ny].first, yz=inc[y][ny].second;
                if (z >= y) break;
                if (neighx[z]==-1) continue;
                int xz=neighx[z];
                neigh[nn]=z;
                neigh_edges[nn]={xz, yz};
                nn++;
            }
            for (int i=0;i<nn;i++) {
                int z = neigh[i], xz = neigh_edges[i].first, yz = neigh_edges[i].second;
                nn2 = 0;
                for (int j=i+1;j<nn;j++) {
                    int w = neigh[j], xw = neigh_edges[j].first, yw = neigh_edges[j].second;
                    if (adjacent(z,w)) {
                        neigh2[nn2]=w;
                        int zw=getEdgeId(z,w);
                        neigh2_edges[nn2]={xw,yw,zw};
                        nn2++;
                    }
                }
                for (int i2=0;i2<nn2;i2++) {
                    int z2 = neigh2[i2];
                    int z2x=neigh2_edges[i2].first, z2y=neigh2_edges[i2].second, z2z=neigh2_edges[i2].third;
                    for (int j2=i2+1;j2<nn2;j2++) {
                        int z3 = neigh2[j2];
                        int z3x=neigh2_edges[j2].first, z3y=neigh2_edges[j2].second, z3z=neigh2_edges[j2].third;
                        if (adjacent(z2,z3)) {
                            int zid=getEdgeId(z2,z3);
                            C5[xy]++; C5[xz]++; C5[yz]++;
                            C5[z2x]++; C5[z2y]++; C5[z2z]++;
                            C5[z3x]++; C5[z3y]++; C5[z3z]++;
                            C5[zid]++;
                        }
                    }
                }
            }
        }
        for (int nx=0;nx<deg[x];nx++) {
            int y=inc[x][nx].first, xy=inc[x][nx].second;
            neighx[y]=-1;
        }
    }
    endTime = clock();
    printf("%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);
    startTime = endTime;

    // set up a system of equations relating orbits for every node
    printf("stage 3 - building systems of equations\n");
    int *common_x = (int*)calloc(n,sizeof(int));
    int *common_x_list = (int*)malloc(n*sizeof(int)), nc_x=0;
    int *common_y = (int*)calloc(n,sizeof(int));
    int *common_y_list = (int*)malloc(n*sizeof(int)), nc_y=0;
    frac_prev=-1;

    for (int x=0;x<n;x++) {
        frac = 100LL*x/n;
        if (frac!=frac_prev) {
            printf("%d%%\r",frac);
            frac_prev=frac;
        }

        // common nodes of x and some other node
        for (int i=0;i<nc_x;i++) common_x[common_x_list[i]]=0;
        nc_x=0;
        for (int nx=0;nx<deg[x];nx++) {
            int a=adj[x][nx];
            for (int na=0;na<deg[a];na++) {
                int z=adj[a][na];
                if (z==x) continue;
                if (common_x[z]==0) common_x_list[nc_x++]=z;
                common_x[z]++;
            }
        }

        for (int nx=0;nx<deg[x];nx++) {
            int y=inc[x][nx].first, xy=inc[x][nx].second;
            int e=xy;
            if (y>=x) break;

            // common nodes of y and some other node
            for (int i=0;i<nc_y;i++) common_y[common_y_list[i]]=0;
            nc_y=0;
            for (int ny=0;ny<deg[y];ny++) {
                int a=adj[y][ny];
                for (int na=0;na<deg[a];na++) {
                    int z=adj[a][na];
                    if (z==y) continue;
                    if (common_y[z]==0) common_y_list[nc_y++]=z;
                    common_y[z]++;
                }
            }

            int64 f_66=0, f_65=0, f_62=0, f_61=0, f_60=0, f_51=0, f_50=0; // 11
            int64 f_64=0, f_58=0, f_55=0, f_48=0, f_41=0, f_35=0; // 10
            int64 f_63=0, f_59=0, f_57=0, f_54=0, f_53=0, f_52=0, f_47=0, f_40=0, f_39=0, f_34=0, f_33=0; // 9
            int64 f_45=0, f_36=0, f_26=0, f_23=0, f_19=0; // 7
            int64 f_49=0, f_38=0, f_37=0, f_32=0, f_25=0, f_22=0, f_18=0; // 6
            int64 f_56=0, f_46=0, f_44=0, f_43=0, f_42=0, f_31=0, f_30=0; // 5
            int64 f_27=0, f_17=0, f_15=0; // 4
            int64 f_20=0, f_16=0, f_13=0; // 3
            int64 f_29=0, f_28=0, f_24=0, f_21=0, f_14=0, f_12=0; // 2

            // smaller (3-node) graphlets
            orbit[x][0] = deg[x];
            for (int nx1=0;nx1<deg[x];nx1++) {
                int z=adj[x][nx1];
                if (z==y) continue;
                if (adjacent(y,z)) eorbit[e][1]++;
                else eorbit[e][0]++;
            }
            for (int ny=0;ny<deg[y];ny++) {
                int z=adj[y][ny];
                if (z==x) continue;
                if (!adjacent(x,z)) eorbit[e][0]++;
            }

            // edge-orbit 11 = (14,14)
            for (int nx1=0;nx1<deg[x];nx1++) {
                int a=adj[x][nx1], xa=inc[x][nx1].second;
                if (a==y || !adjacent(y,a)) continue;
                for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                    int b=adj[x][nx2], xb=inc[x][nx2].second;
                    if (b==y || !adjacent(y,b) || !adjacent(a,b)) continue;
                    int ya=getEdgeId(y,a), yb=getEdgeId(y,b), ab=getEdgeId(a,b);
                    eorbit[e][11]++;
                    f_66 += common3_get(TRIPLE(x,y,a))-1;
                    f_66 += common3_get(TRIPLE(x,y,b))-1;
                    f_65 += common3_get(TRIPLE(a,b,x))-1;
                    f_65 += common3_get(TRIPLE(a,b,y))-1;
                    f_62 += tri[xy]-2;
                    f_61 += (tri[xa]-2)+(tri[xb]-2)+(tri[ya]-2)+(tri[yb]-2);
                    f_60 += tri[ab]-2;
                    f_51 += (deg[x]-3)+(deg[y]-3);
                    f_50 += (deg[a]-3)+(deg[b]-3);
                }
            }

            // edge-orbit 10 = (13,13)
            for (int nx1=0;nx1<deg[x];nx1++) {
                int a=adj[x][nx1], xa=inc[x][nx1].second;
                if (a==y || !adjacent(y,a)) continue;
                for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                    int b=adj[x][nx2], xb=inc[x][nx2].second;
                    if (b==y || !adjacent(y,b) || adjacent(a,b)) continue;
                    int ya=getEdgeId(y,a), yb=getEdgeId(y,b);
                    eorbit[e][10]++;
                    f_64 += common3_get(TRIPLE(a,b,x))-1;
                    f_64 += common3_get(TRIPLE(a,b,y))-1;
                    f_58 += common2_get(PAIR(a,b))-2;
                    f_55 += (tri[xa]-1)+(tri[xb]-1)+(tri[ya]-1)+(tri[yb]-1);
                    f_48 += tri[xy]-2;
                    f_41 += (deg[a]-2)+(deg[b]-2);
                    f_35 += (deg[x]-3)+(deg[y]-3);
                }
            }

            // edge-orbit 9 = (12,13)
            for (int nx=0;nx<deg[x];nx++) {
                int a=adj[x][nx], xa=inc[x][nx].second;
                if (a==y) continue;
                for (int ny=0;ny<deg[y];ny++) {
                    int b=adj[y][ny], yb=inc[y][ny].second;
                    if (b==x || !adjacent(a,b)) continue;
                    int adj_ya=adjacent(y,a), adj_xb=adjacent(x,b);
                    if (adj_ya+adj_xb!=1) continue;
                    int ab=getEdgeId(a,b);
                    eorbit[e][9]++;
                    if (adj_xb) {
                        int xb=getEdgeId(x,b);
                        f_63 += common3_get(TRIPLE(a,b,y))-1;
                        f_59 += common3_get(TRIPLE(a,b,x));
                        f_57 += common_y[a]-2;
                        f_54 += tri[yb]-1;
                        f_53 += tri[xa]-1;
                        f_47 += tri[xb]-2;
                        f_40 += deg[y]-2;
                        f_39 += deg[a]-2;
                        f_34 += deg[x]-3;
                        f_33 += deg[b]-3;
                    } else if (adj_ya) {
                        int ya=getEdgeId(y,a);
                        f_63 += common3_get(TRIPLE(a,b,x))-1;
                        f_59 += common3_get(TRIPLE(a,b,y));
                        f_57 += common_x[b]-2;
                        f_54 += tri[xa]-1;
                        f_53 += tri[yb]-1;
                        f_47 += tri[ya]-2;
                        f_40 += deg[x]-2;
                        f_39 += deg[b]-2;
                        f_34 += deg[y]-3;
                        f_33 += deg[a]-3;
                    }
                    f_52 += tri[ab]-1;
                }
            }

            // edge-orbit 8 = (10,11)
            for (int nx=0;nx<deg[x];nx++) {
                int a=adj[x][nx];
                if (a==y || !adjacent(y,a)) continue;
                for (int nx1=0;nx1<deg[x];nx1++) {
                    int b=adj[x][nx1];
                    if (b==y || b==a || adjacent(y,b) || adjacent(a,b)) continue;
                    eorbit[e][8]++;
                }
                for (int ny1=0;ny1<deg[y];ny1++) {
                    int b=adj[y][ny1];
                    if (b==x || b==a || adjacent(x,b) || adjacent(a,b)) continue;
                    eorbit[e][8]++;
                }
            }

            // edge-orbit 7 = (10,10)
            for (int nx=0;nx<deg[x];nx++) {
                int a=adj[x][nx];
                if (a==y || !adjacent(y,a)) continue;
                for (int na=0;na<deg[a];na++) {
                    int b=adj[a][na], ab=inc[a][na].second;
                    if (b==x || b==y || adjacent(x,b) || adjacent(y,b)) continue;
                    eorbit[e][7]++;
                    f_45 += common_x[b]-1;
                    f_45 += common_y[b]-1;
                    f_36 += tri[ab];
                    f_26 += deg[a]-3;
                    f_23 += deg[b]-1;
                    f_19 += (deg[x]-2)+(deg[y]-2);
                }
            }

            // edge-orbit 6 = (9,11)
            for (int ny1=0;ny1<deg[y];ny1++) {
                int a=adj[y][ny1], ya=inc[y][ny1].second;
                if (a==x || adjacent(x,a)) continue;
                for (int ny2=ny1+1;ny2<deg[y];ny2++) {
                    int b=adj[y][ny2], yb=inc[y][ny2].second;
                    if (b==x || adjacent(x,b) || !adjacent(a,b)) continue;
                    int ab=getEdgeId(a,b);
                    eorbit[e][6]++;
                    f_49 += common3_get(TRIPLE(y,a,b));
                    f_38 += tri[ab]-1;
                    f_37 += tri[xy];
                    f_32 += (tri[ya]-1)+(tri[yb]-1);
                    f_25 += deg[y]-3;
                    f_22 += deg[x]-1;
                    f_18 += (deg[a]-2)+(deg[b]-2);
                }
            }
            for (int nx1=0;nx1<deg[x];nx1++) {
                int a=adj[x][nx1], xa=inc[x][nx1].second;
                if (a==y || adjacent(y,a)) continue;
                for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                    int b=adj[x][nx2], xb=inc[x][nx2].second;
                    if (b==y || adjacent(y,b) || !adjacent(a,b)) continue;
                    int ab=getEdgeId(a,b);
                    eorbit[e][6]++;
                    f_49 += common3_get(TRIPLE(x,a,b));
                    f_38 += tri[ab]-1;
                    f_37 += tri[xy];
                    f_32 += (tri[xa]-1)+(tri[xb]-1);
                    f_25 += deg[x]-3;
                    f_22 += deg[y]-1;
                    f_18 += (deg[a]-2)+(deg[b]-2);
                }
            }

            // edge-orbit 5 = (8,8)
            for (int nx=0;nx<deg[x];nx++) {
                int a=adj[x][nx], xa=inc[x][nx].second;
                if (a==y || adjacent(y,a)) continue;
                for (int ny=0;ny<deg[y];ny++) {
                    int b=adj[y][ny], yb=inc[y][ny].second;
                    if (b==x || adjacent(x,b) || !adjacent(a,b)) continue;
                    int ab=getEdgeId(a,b);
                    eorbit[e][5]++;
                    f_56 += common3_get(TRIPLE(x,a,b));
                    f_56 += common3_get(TRIPLE(y,a,b));
                    f_46 += tri[xy];
                    f_44 += tri[xa]+tri[yb];
                    f_43 += tri[ab];
                    f_42 += common_x[b]-2;
                    f_42 += common_y[a]-2;
                    f_31 += (deg[x]-2)+(deg[y]-2);
                    f_30 += (deg[a]-2)+(deg[b]-2);
                }
            }

            // edge-orbit 4 = (6,7)
            for (int ny1=0;ny1<deg[y];ny1++) {
                int a=adj[y][ny1];
                if (a==x || adjacent(x,a)) continue;
                for (int ny2=ny1+1;ny2<deg[y];ny2++) {
                    int b=adj[y][ny2];
                    if (b==x || adjacent(x,b) || adjacent(a,b)) continue;
                    eorbit[e][4]++;
                    f_27 += tri[xy];
                    f_17 += deg[y]-3;
                    f_15 += (deg[a]-1)+(deg[b]-1);
                }
            }
            for (int nx1=0;nx1<deg[x];nx1++) {
                int a=adj[x][nx1];
                if (a==y || adjacent(y,a)) continue;
                for (int nx2=nx1+1;nx2<deg[x];nx2++) {
                    int b=adj[x][nx2];
                    if (b==y || adjacent(y,b) || adjacent(a,b)) continue;
                    eorbit[e][4]++;
                    f_27 += tri[xy];
                    f_17 += deg[x]-3;
                    f_15 += (deg[a]-1)+(deg[b]-1);
                }
            }

            // edge-orbit 3 = (5,5)
            for (int nx=0;nx<deg[x];nx++) {
                int a=adj[x][nx];
                if (a==y || adjacent(y,a)) continue;
                for (int ny=0;ny<deg[y];ny++) {
                    int b=adj[y][ny];
                    if (b==x || adjacent(x,b) || adjacent(a,b)) continue;
                    eorbit[e][3]++;
                    f_20 += tri[xy];
                    f_16 += (deg[x]-2)+(deg[y]-2);
                    f_13 += (deg[a]-1)+(deg[b]-1);
                }
            }

            // edge-orbit 2 = (4,5)
            for (int ny=0;ny<deg[y];ny++) {
                int a=adj[y][ny];
                if (a==x || adjacent(x,a)) continue;
                for (int na=0;na<deg[a];na++) {
                    int b=adj[a][na], ab=inc[a][na].second;
                    if (b==y || adjacent(y,b) || adjacent(x,b)) continue;
                    eorbit[e][2]++;
                    f_29 += common_y[b]-1;
                    f_28 += common_x[b];
                    f_24 += tri[xy];
                    f_21 += tri[ab];
                    f_14 += deg[a]-2;
                    f_12 += deg[b]-1;
                }
            }
            for (int nx=0;nx<deg[x];nx++) {
                int a=adj[x][nx];
                if (a==y || adjacent(y,a)) continue;
                for (int na=0;na<deg[a];na++) {
                    int b=adj[a][na], ab=inc[a][na].second;
                    if (b==x || adjacent(x,b) || adjacent(y,b)) continue;
                    eorbit[e][2]++;
                    f_29 += common_x[b]-1;
                    f_28 += common_y[b];
                    f_24 += tri[xy];
                    f_21 += tri[ab];
                    f_14 += deg[a]-2;
                    f_12 += deg[b]-1;
                }
            }

            // solve system of equations
            eorbit[e][67]=C5[e];
            eorbit[e][66]=(f_66-6*eorbit[e][67])/2;
            eorbit[e][65]=(f_65-6*eorbit[e][67]);
            eorbit[e][64]=(f_64-2*eorbit[e][66]);
            eorbit[e][63]=(f_63-2*eorbit[e][65])/2;
            eorbit[e][62]=(f_62-2*eorbit[e][66]-3*eorbit[e][67]);
            eorbit[e][61]=(f_61-2*eorbit[e][65]-4*eorbit[e][66]-12*eorbit[e][67]);
            eorbit[e][60]=(f_60-1*eorbit[e][65]-3*eorbit[e][67]);
            eorbit[e][59]=(f_59-2*eorbit[e][65])/2;
            eorbit[e][58]=(f_58-1*eorbit[e][64]-1*eorbit[e][66]);
            eorbit[e][57]=(f_57-2*eorbit[e][63]-2*eorbit[e][64]-2*eorbit[e][65]);
            eorbit[e][56]=(f_56-2*eorbit[e][63])/2;
            eorbit[e][55]=(f_55-4*eorbit[e][62]-2*eorbit[e][64]-4*eorbit[e][66]);
            eorbit[e][54]=(f_54-1*eorbit[e][61]-2*eorbit[e][63]-2*eorbit[e][65])/2;
            eorbit[e][53]=(f_53-2*eorbit[e][59]-2*eorbit[e][64]-2*eorbit[e][65]);
            eorbit[e][52]=(f_52-2*eorbit[e][59]-2*eorbit[e][63]-2*eorbit[e][65]);
            eorbit[e][51]=(f_51-1*eorbit[e][61]-2*eorbit[e][62]-1*eorbit[e][65]-4*eorbit[e][66]-6*eorbit[e][67]);
            eorbit[e][50]=(f_50-2*eorbit[e][60]-1*eorbit[e][61]-2*eorbit[e][65]-2*eorbit[e][66]-6*eorbit[e][67]);
            eorbit[e][49]=(f_49-1*eorbit[e][59])/3;
            eorbit[e][48]=(f_48-2*eorbit[e][62]-1*eorbit[e][66])/3;
            eorbit[e][47]=(f_47-2*eorbit[e][59]-1*eorbit[e][61]-2*eorbit[e][65])/2;
            eorbit[e][46]=(f_46-1*eorbit[e][57]-1*eorbit[e][63]);
            eorbit[e][45]=(f_45-1*eorbit[e][52]-4*eorbit[e][58]-4*eorbit[e][60]);
            eorbit[e][44]=(f_44-2*eorbit[e][56]-1*eorbit[e][57]-2*eorbit[e][63]);
            eorbit[e][43]=(f_43-2*eorbit[e][56]-1*eorbit[e][63]);
            eorbit[e][42]=(f_42-2*eorbit[e][56]-1*eorbit[e][57]-2*eorbit[e][63])/2;
            eorbit[e][41]=(f_41-1*eorbit[e][55]-2*eorbit[e][58]-2*eorbit[e][62]-2*eorbit[e][64]-2*eorbit[e][66]);
            eorbit[e][40]=(f_40-2*eorbit[e][54]-1*eorbit[e][55]-1*eorbit[e][57]-1*eorbit[e][61]-2*eorbit[e][63]-2*eorbit[e][64]-2*eorbit[e][65]);
            eorbit[e][39]=(f_39-1*eorbit[e][52]-1*eorbit[e][53]-1*eorbit[e][57]-2*eorbit[e][59]-2*eorbit[e][63]-2*eorbit[e][64]-2*eorbit[e][65]);
            eorbit[e][38]=(f_38-3*eorbit[e][49]-1*eorbit[e][56]-1*eorbit[e][59]);
            eorbit[e][37]=(f_37-1*eorbit[e][53]-1*eorbit[e][59]);
            eorbit[e][36]=(f_36-1*eorbit[e][52]-2*eorbit[e][60])/2;
            eorbit[e][35]=(f_35-6*eorbit[e][48]-1*eorbit[e][55]-4*eorbit[e][62]-1*eorbit[e][64]-2*eorbit[e][66]);
            eorbit[e][34]=(f_34-2*eorbit[e][47]-1*eorbit[e][53]-1*eorbit[e][55]-2*eorbit[e][59]-1*eorbit[e][61]-2*eorbit[e][64]-2*eorbit[e][65]);
            eorbit[e][33]=(f_33-2*eorbit[e][47]-1*eorbit[e][52]-2*eorbit[e][54]-2*eorbit[e][59]-1*eorbit[e][61]-2*eorbit[e][63]-2*eorbit[e][65]);
            eorbit[e][32]=(f_32-6*eorbit[e][49]-1*eorbit[e][53]-2*eorbit[e][59])/2;
            eorbit[e][31]=(f_31-2*eorbit[e][42]-1*eorbit[e][44]-2*eorbit[e][46]-2*eorbit[e][56]-2*eorbit[e][57]-2*eorbit[e][63]);
            eorbit[e][30]=(f_30-2*eorbit[e][42]-2*eorbit[e][43]-1*eorbit[e][44]-4*eorbit[e][56]-1*eorbit[e][57]-2*eorbit[e][63]);
            eorbit[e][29]=(f_29-2*eorbit[e][38]-1*eorbit[e][45]-1*eorbit[e][52])/2;
            eorbit[e][28]=(f_28-2*eorbit[e][43]-1*eorbit[e][45]-1*eorbit[e][52])/2;
            eorbit[e][27]=(f_27-1*eorbit[e][34]-1*eorbit[e][47]);
            eorbit[e][26]=(f_26-1*eorbit[e][33]-2*eorbit[e][36]-1*eorbit[e][50]-1*eorbit[e][52]-2*eorbit[e][60])/2;
            eorbit[e][25]=(f_25-2*eorbit[e][32]-1*eorbit[e][37]-3*eorbit[e][49]-1*eorbit[e][53]-1*eorbit[e][59]);
            eorbit[e][24]=(f_24-1*eorbit[e][39]-1*eorbit[e][45]-1*eorbit[e][52]);
            eorbit[e][23]=(f_23-2*eorbit[e][36]-1*eorbit[e][45]-1*eorbit[e][52]-2*eorbit[e][58]-2*eorbit[e][60]);
            eorbit[e][22]=(f_22-1*eorbit[e][37]-1*eorbit[e][44]-1*eorbit[e][53]-1*eorbit[e][56]-1*eorbit[e][59]);
            eorbit[e][21]=(f_21-2*eorbit[e][38]-2*eorbit[e][43]-1*eorbit[e][52])/2;
            eorbit[e][20]=(f_20-1*eorbit[e][40]-1*eorbit[e][54]);
            eorbit[e][19]=(f_19-1*eorbit[e][33]-2*eorbit[e][41]-1*eorbit[e][45]-2*eorbit[e][50]-1*eorbit[e][52]-4*eorbit[e][58]-4*eorbit[e][60]);
            eorbit[e][18]=(f_18-2*eorbit[e][32]-2*eorbit[e][38]-1*eorbit[e][44]-6*eorbit[e][49]-1*eorbit[e][53]-2*eorbit[e][56]-2*eorbit[e][59]);
            eorbit[e][17]=(f_17-2*eorbit[e][25]-1*eorbit[e][27]-1*eorbit[e][32]-1*eorbit[e][34]-1*eorbit[e][47])/3;
            eorbit[e][16]=(f_16-2*eorbit[e][20]-2*eorbit[e][22]-1*eorbit[e][31]-2*eorbit[e][40]-1*eorbit[e][44]-2*eorbit[e][54])/2;
            eorbit[e][15]=(f_15-2*eorbit[e][25]-2*eorbit[e][29]-1*eorbit[e][31]-2*eorbit[e][32]-1*eorbit[e][34]-2*eorbit[e][42]-2*eorbit[e][47]);
            eorbit[e][14]=(f_14-1*eorbit[e][18]-2*eorbit[e][21]-1*eorbit[e][30]-2*eorbit[e][38]-1*eorbit[e][39]-2*eorbit[e][43]-1*eorbit[e][52])/2;
            eorbit[e][13]=(f_13-2*eorbit[e][22]-2*eorbit[e][28]-1*eorbit[e][31]-1*eorbit[e][40]-2*eorbit[e][44]-2*eorbit[e][54]);
            eorbit[e][12]=(f_12-2*eorbit[e][21]-2*eorbit[e][28]-2*eorbit[e][29]-2*eorbit[e][38]-2*eorbit[e][43]-1*eorbit[e][45]-1*eorbit[e][52]);
        }
    }

    endTime = clock();
    printf("%.2f\n", (double)(endTime-startTime)/CLOCKS_PER_SEC);

    endTime_all = endTime;
    printf("total: %.2f\n", (double)(endTime_all-startTime_all)/CLOCKS_PER_SEC);
} 

int writeResults(int g, const char* output_filename) {
    fstream fout; 
    if (fout.fail()) {
        cerr << "Failed to open file " << output_filename << endl;
        return 1;
    }
    fout.open(output_filename, fstream::out | fstream::binary);
    int no[] = {0,0,1,4,15,73};
    for (int i=0;i<n;i++) {
        for (int j=0;j<no[g];j++) {
            if (j!=0) 
                fout << " ";
            fout << orbit[i][j];
        }
        fout << endl;
    }
    fout.close();
}

string writeResultsString(int g) {
    std::stringstream ss("", ios_base::app | ios_base::out);
    int no[] = {0,0,1,4,15,73};
    for (int i=0;i<n;i++) {
        for (int j=0;j<no[g];j++) {
            if (j!=0) 
                ss << " ";
            ss << orbit[i][j];
        }
        ss << endl;
    }
    return ss.str();
}

int writeEdgeResults(int g, const char* output_filename) {
    fstream fout; 
    if (fout.fail()) {
        cerr << "Failed to open file " << output_filename << endl;
        return 1;
    }
    int no[] = {0,0,0,2,12,68};
    for (int i=0;i<m;i++) {
        for (int j=0;j<no[g];j++) {
            if (j!=0) fout << " ";
            fout << eorbit[i][j];
        }
        fout << endl;
    }
    fout.close();
}

string writeEdgeResultsString(int g) {
    std::stringstream ss("", ios_base::app | ios_base::out);
    int no[] = {0,0,0,2,12,68};
    for (int i=0;i<m;i++) {
        for (int j=0;j<no[g];j++) {
            if (j!=0) ss << " ";
            ss << eorbit[i][j];
        }
        ss << endl;
    }
    return ss.str();
}

int motif_counts(const char* orbit_type, int graphlet_size, 
        const char* input_filename, const char* output_filename, string &out_str) {
    fstream fin; // input and output files
    // open input, output files
    if (strcmp(orbit_type, "node")!=0 && strcmp(orbit_type, "edge")!=0) {
        cerr << "Incorrect orbit type '" << orbit_type << "'. Should be 'node' or 'edge'." << endl;
        return 0;
    }
    if (graphlet_size!=4 && graphlet_size!=5) {
        cerr << "Incorrect graphlet size " << graphlet_size << ". Should be 4 or 5." << endl;
        return 0;
    }
    fin.open(input_filename, fstream::in);
    if (fin.fail()) {
        cerr << "Failed to open file " << input_filename << endl;
        return 0;
    }
    // read input graph
    fin >> n >> m;
    int d_max=0;
    edges = (PAIR*)malloc(m*sizeof(PAIR));
    deg = (int*)calloc(n,sizeof(int));
    for (int i=0;i<m;i++) {
        int a,b;
        fin >> a >> b;
        if (!(0<=a && a<n) || !(0<=b && b<n)) {
            cerr << "Node ids should be between 0 and n-1." << endl;
            return 0;
        }
        if (a==b) {
            cerr << "Self loops (edge from x to x) are not allowed." << endl;
            return 0;
        }
        deg[a]++; deg[b]++;
        edges[i]=PAIR(a,b);
    }
    for (int i=0;i<n;i++) d_max=max(d_max,deg[i]);
    printf("nodes: %d\n",n);
    printf("edges: %d\n",m);
    printf("max degree: %d\n",d_max);
    fin.close();
    if ((int)(set<PAIR>(edges,edges+m).size())!=m) {
        cerr << "Input file contains duplicate undirected edges." << endl;
        return 0;
    }
    // set up adjacency matrix if it's smaller than 100MB
    if ((int64)n*n < 100LL*1024*1024*8) {
        adjacent = adjacent_matrix;
        adj_matrix = (int*)calloc((n*n)/adj_chunk+1,sizeof(int));
        for (int i=0;i<m;i++) {
            int a=edges[i].a, b=edges[i].b;
            adj_matrix[(a*n+b)/adj_chunk]|=(1<<((a*n+b)%adj_chunk));
            adj_matrix[(b*n+a)/adj_chunk]|=(1<<((b*n+a)%adj_chunk));
        }
    } else {
        adjacent = adjacent_list;
    }
    // set up adjacency, incidence lists
    adj = (int**)malloc(n*sizeof(int*));
    for (int i=0;i<n;i++) adj[i] = (int*)malloc(deg[i]*sizeof(int));
    inc = (PII**)malloc(n*sizeof(PII*));
    for (int i=0;i<n;i++) inc[i] = (PII*)malloc(deg[i]*sizeof(PII));
    int *d = (int*)calloc(n,sizeof(int));
    for (int i=0;i<m;i++) {
        int a=edges[i].a, b=edges[i].b;
        adj[a][d[a]]=b; adj[b][d[b]]=a;
        inc[a][d[a]]=PII(b,i); inc[b][d[b]]=PII(a,i);
        d[a]++; d[b]++;
    }
    for (int i=0;i<n;i++) {
        sort(adj[i],adj[i]+deg[i]);
        sort(inc[i],inc[i]+deg[i]);
    }
    // initialize orbit counts
    orbit = (int64**)malloc(n*sizeof(int64*));
    for (int i=0;i<n;i++) orbit[i] = (int64*)calloc(73,sizeof(int64));
    // initialize edge orbit counts
    eorbit = (int64**)malloc(m*sizeof(int64*));
    for (int i=0;i<m;i++) eorbit[i] = (int64*)calloc(68,sizeof(int64));

    if (strcmp(orbit_type,"node") == 0) {
        printf("Counting NODE orbits of graphlets on %d nodes.\n\n",graphlet_size);
        if (graphlet_size==4) count4();
        if (graphlet_size==5) count5();
        if (strcmp(output_filename, "std") == 0) {
            cout << "orbit counts: \n" << writeResultsString(graphlet_size) << endl;
        } else {
            out_str = writeResults(graphlet_size, output_filename);
        }
    } else {
        printf("Counting EDGE orbits of graphlets on %d nodes.\n\n",graphlet_size);
        if (graphlet_size==4) ecount4();
        if (graphlet_size==5) ecount5();
        if (strcmp(output_filename, "std") == 0) {
            cout << "orbit counts: \n" << writeEdgeResultsString(graphlet_size) << endl;
        } else {
            out_str = writeEdgeResults(graphlet_size, output_filename);
        }
    }

    return 1;
}

int init(int argc, char *argv[]) {
    if (argc!=5) {
        cerr << "Incorrect number of arguments." << endl;
        cerr << "Usage: orca.exe [orbit type: node|edge] [graphlet size: 4/5] [graph - input file] [graphlets - output file]" << endl;
        return 0;
    }
    int graphlet_size;
    sscanf(argv[2],"%d", &graphlet_size);
    string out;
    motif_counts(argv[1], graphlet_size, argv[3], argv[4], out);

    return 1;
}


int main(int argc, char *argv[]) {


    if (!init(argc, argv)) {
//        cerr << "Stopping!" << endl;
        return 1;
    }
    

    return 0;
}

