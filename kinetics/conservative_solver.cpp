#include <algorithm>
#include <cmath>
static void rates(const double* y, const double* logk, const double* q,
                  const long long* ri, const double* ref, int nr, double* r) {
    for (int j=0; j<nr; ++j) {
        const int a=ri[2*j], b=ri[2*j+1];
        r[j]=0.;
        if (y[a]>0. && y[b]>0.) {
            double lograte=logk[j]+q[5*j+3]*std::log(y[a]/ref[3*j+1])
                          +q[5*j+4]*std::log(y[b]/ref[3*j+2]);
            r[j]=std::exp(std::clamp(lograte,-80.,80.));
        }
    }
}

extern "C" void integrate_conservative(
    const double* q, const double* initial, const double* times,
    const double* env, const double* s, const long long* ri, const double* ref,
    double tref, double max_step, int ne, int nt, int ns, int nr, double* out) {
    for (int e=0; e<ne; ++e) {
        double y[4], logk[2];
        for (int i=0;i<ns;++i) out[(e*nt)*ns+i]=y[i]=initial[e*ns+i];
        for (int j=0;j<nr;++j)
            logk[j]=q[5*j]-q[5*j+1]*100000./8.314*(1./env[2*e]-1./tref)
                    +q[5*j+2]*std::log(env[2*e+1]/ref[3*j]);
        for (int p=1;p<nt;++p) {
            const double interval=times[e*nt+p]-times[e*nt+p-1];
            const int n=std::max(1,(int)std::ceil(interval/max_step));
            const double dt=interval/n;
            for (int step=0;step<n;++step) {
                double r[4][2], tmp[4];
                rates(y,logk,q,ri,ref,nr,r[0]);
                for (int stage=1;stage<4;++stage) {
                    for (int i=0;i<ns;++i) {
                        double dy=0.;
                        for (int j=0;j<nr;++j) dy+=s[i*nr+j]*r[stage-1][j];
                        tmp[i]=y[i]+dt*(stage==3?1.:.5)*dy;
                    }
                    rates(tmp,logk,q,ri,ref,nr,r[stage]);
                }
                double flux[2], change[4], factor=1.;
                for (int j=0;j<nr;++j)
                    flux[j]=dt*(r[0][j]+2*r[1][j]+2*r[2][j]+r[3][j])/6.;
                for (int i=0;i<ns;++i) {
                    change[i]=0.;
                    for (int j=0;j<nr;++j) change[i]+=s[i*nr+j]*flux[j];
                    if (change[i]<0.) factor=std::min(factor,std::max(0.,y[i])/-change[i]);
                }
                for (int i=0;i<ns;++i) y[i]+=factor*change[i];
            }
            for (int i=0;i<ns;++i) out[(e*nt+p)*ns+i]=y[i];
        }
    }
}
