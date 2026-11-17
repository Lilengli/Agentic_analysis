#include "../fused_kernel.hpp"
#include <cstdio>
#include <random>
#include <stdexcept>

template <unsigned BM, unsigned BN>
void check(unsigned seq, unsigned heads, unsigned cores, bool tail_only=false) {
    std::mt19937 gen(1234+seq);
    std::normal_distribution<float> rand(0,1);
    const size_t count=size_t(heads)*seq*128;
    std::vector<half> q(count),k(count),v(count),out(count);
    for(auto* tensor:{&q,&k,&v}) for(auto& x:*tensor) x=half(rand(gen));
    std::fill(out.begin(),out.end(),half(NAN));
    const auto original_q=q,original_k=k,original_v=v;
    AscendC::block_num=cores;
    AscendC::gm_read_elements=AscendC::gm_write_elements=0;
    unsigned first_core=tail_only?cores-1:0;
    for(unsigned core=first_core;core<cores;++core) {
        AscendC::block_idx=core;
        FusedAiv::fused_causal_attention<BM,BN>((uint8_t*)q.data(),(uint8_t*)k.data(),
            (uint8_t*)v.data(),(uint8_t*)out.data(),heads,seq);
    }
    if(AscendC::gm_write_elements!=(tail_only?size_t(BM)*128:count)) throw std::runtime_error("unexpected GM write count");
    if(q!=original_q || k!=original_k || v!=original_v) throw std::runtime_error("input modified");
    double error=0;
    for(unsigned head=0;head<heads;++head) for(unsigned row=0;row<seq;++row) {
        if(tail_only && (head!=heads-1 || row<seq-BM)) continue;
        size_t base=size_t(head)*seq*128;
        std::vector<double> scores(row+1), acc(128,0);
        double max=-INFINITY;
        for(unsigned j=0;j<=row;++j) {
            double dot=0;
            for(unsigned d=0;d<128;++d) dot+=double(q[base+row*128+d])*double(k[base+j*128+d]);
            scores[j]=dot/std::sqrt(128.0);
            max=std::max(max,scores[j]);
        }
        double l=0;
        for(unsigned j=0;j<=row;++j) {
            const double p=std::exp(scores[j]-max); l+=p;
            for(unsigned d=0;d<128;++d) acc[d]+=p*double(v[base+j*128+d]);
        }
        for(unsigned d=0;d<128;++d) {
            double expected=acc[d]/l;
            double got=double(out[base+row*128+d]);
            double diff=std::abs(got-expected); error=std::max(error,diff);
            if(!std::isfinite(got) || diff>0.002+0.01*std::abs(expected))
                throw std::runtime_error("reference comparison failed");
        }
    }
    std::printf("PASS BM=%u BN=%u S=%u H=%u cores=%u tail_only=%d max_abs=%.8g UB=%u GM_writes=%llu\n",
        BM,BN,seq,heads,cores,tail_only,error,FusedAiv::UbBytes<BM,BN>(),
        (unsigned long long)AscendC::gm_write_elements);
}

int main() {
    for(unsigned seq:{64,128,256}) for(unsigned heads:{1,2}) {
        check<8,32>(seq,heads,7); check<16,32>(seq,heads,48);
        check<8,64>(seq,heads,48); check<16,64>(seq,heads,7);
    }
    // Exercise 16K offsets, last-head selection and all KV iterations without
    // pretending the CPU simulator validated every row of a full 16K result.
    check<8,32>(16384,2,2*16384/8,true);
    check<16,32>(16384,2,2*16384/16,true);
    check<8,64>(16384,2,2*16384/8,true);
    check<16,64>(16384,2,2*16384/16,true);
}
