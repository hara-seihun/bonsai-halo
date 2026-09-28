#include "engine.h"
#include "tokenizer.h"
#include "speculative_recycle.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace halo;
static int diagnostic_offset=-1;
static double clock_s(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double sync_s(Engine &e){auto code=hipStreamSynchronize(e.stream);if(code!=hipSuccess)throw std::runtime_error(hipGetErrorString(code));return clock_s();}
struct Copy {std::vector<int> tokens; int matched=0;};
static Copy proposal(const std::vector<int>& visible,int pending){
    auto history=visible; history.push_back(pending); Copy best;
    for(int end=(int)visible.size()-1;end>=0;--end){
        if(history[end]!=pending)continue;
        int n=1; while(n<32 && end-n>=0 && (int)history.size()-1-n>=0 && history[end-n]==history[history.size()-1-n])++n;
        if(n<=best.matched)continue;
        int length=std::min(7,(int)visible.size()-end-1);
        if(length<=0)continue;
        best.matched=n;best.tokens.assign(history.begin()+end+1,history.begin()+end+1+length);
    }
    return best;
}
struct Result {std::vector<int> tokens; int next=-1; long steps=0,nodes=0,copy_steps=0,df_steps=0,recycle_steps=0; double prefill=0,decode=0,draft=0,verify=0; std::vector<int> progress;};
static Result run(Engine &e,Tokenizer &tk,const std::vector<int>&prompt,int limit,const std::string&method){
    const bool neural=method!="serial" && method!="copy";
    e.df.loaded=neural; e.reset();
    Result r; std::vector<int> drafts,am,visible=prompt;int next=-1;
    double t0=sync_s(e);e.prefill(e.seq0,prompt,0,next,neural?&drafts:nullptr);
    if(neural){
        DflashParams p{};p.w=e.df.use_q4?e.df.w4:e.df.w;p.hproj_out=e.df.hproj_out;p.topi=e.df.topi;p.topv=e.df.topv;p.draft_out=e.df.draft_out;
        launch_speculative_recycle(p,1,next,e.stream);
        std::vector<int> check(drafts.size());
        auto error=hipMemcpyAsync(check.data(),e.df.draft_out,check.size()*sizeof(int),hipMemcpyDeviceToHost,e.stream);
        if(error!=hipSuccess)throw std::runtime_error(hipGetErrorString(error));
        sync_s(e);if(check!=drafts)throw std::runtime_error("saved-table selector disagrees with complete drafter selector");
    }
    r.prefill=sync_s(e)-t0;
    e.fwd.hcap=neural?e.hcap:nullptr;
    std::vector<Engine::Seq*> sv{&e.seq0};
    int previous=8, block_origin=e.seq0.len; const double began=sync_s(e);
    while((int)r.tokens.size()<limit && !tk.is_eog(next)){
        double td=sync_s(e);bool copied=false;std::vector<int> chosen;
        if(method=="copy" || method=="hybrid"){
            auto c=proposal(visible,next);
            if(c.matched>=(method=="hybrid"?4:2)){chosen=c.tokens;copied=true;}
        }
        if(!copied && neural)chosen=drafts;
        int width=8;
        if(method=="df4")width=4;
        if(method=="df6")width=6;
        if(method=="adaptive")width=previous<=2?4:8;
        if((int)chosen.size()>width-1)chosen.resize(width-1);
        const int room=limit-(int)r.tokens.size();
        if((int)chosen.size()>room-1)chosen.resize(room-1);
        std::vector<int> block{next};block.insert(block.end(),chosen.begin(),chosen.end());
        const double tv=sync_s(e);r.draft+=tv-td;
        const int old=e.seq0.len;
        e.forward(sv,{block},true,&am);r.verify+=sync_s(e)-tv;
        int accepted=1;
        while(accepted<(int)block.size() && am[accepted-1]==block[accepted] && !tk.is_eog(block[accepted]))++accepted;
        for(int i=0;i<accepted;++i){r.tokens.push_back(block[i]);visible.push_back(block[i]);}
        next=am[accepted-1]; e.rollback(e.seq0,accepted);
        if((int)r.tokens.size()==diagnostic_offset){
            std::vector<float> logits;e.get_logits(logits,accepted-1);
            int first=0,second=1;if(logits[second]>logits[first])std::swap(first,second);
            for(int k=2;k<(int)logits.size();++k){if(logits[k]>logits[first]){second=first;first=k;}else if(logits[k]>logits[second])second=k;}
            unsigned long long hash=1469598103934665603ull;
            for(int token:visible)for(int b=0;b<4;b++){hash^=(unsigned)(token>>(8*b))&255u;hash*=1099511628211ull;}
            fprintf(stderr,"decision method=%s offset=%d prefix=%llu next=%d first=%d:%.9g second=%d:%.9g gap=%.9g\n",method.c_str(),diagnostic_offset,hash,next,first,logits[first],second,logits[second],logits[first]-logits[second]);
        }
        r.steps++;r.nodes+=block.size();r.copy_steps+=copied;r.progress.push_back(accepted);previous=accepted;
        if((int)r.tokens.size()>=limit || tk.is_eog(next))break;
        if(neural){
            const double t=sync_s(e);
            bool next_copy=false;
            if(method=="hybrid")next_copy=proposal(visible,next).matched>=4;
            const int begin=e.seq0.len-block_origin+1;
            const bool recycled=method=="recycle" && begin<=6;
            e.dflash_step(e.seq0,accepted,old,next,e.seq0.len,(next_copy||recycled)?nullptr:&drafts);
            if(recycled){
                DflashParams p{};p.w=e.df.use_q4?e.df.w4:e.df.w;
                p.hproj_out=e.df.hproj_out;p.topi=e.df.topi;p.topv=e.df.topv;p.draft_out=e.df.draft_out;
                launch_speculative_recycle(p,begin,next,e.stream);
                drafts.resize(DF_BLOCK-begin);
                auto error=hipMemcpyAsync(drafts.data(),e.df.draft_out,drafts.size()*sizeof(int),hipMemcpyDeviceToHost,e.stream);
                if(error!=hipSuccess)throw std::runtime_error(hipGetErrorString(error));
                sync_s(e);r.recycle_steps++;
            }else if(!next_copy){r.df_steps++;block_origin=e.seq0.len;}
            r.draft+=sync_s(e)-t;
        }
    }
    r.decode=sync_s(e)-began;r.next=next;e.fwd.hcap=nullptr;e.df.loaded=true;return r;
}
static void array(const std::vector<int>&v){std::cout<<'[';for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
int main(int argc,char**argv){try{
    std::string model="../../data/bonsai2/PTQ1_0.gguf",drafter="../../data/bonsai2/drafters/dflash2.safetensors",file,methods="serial,df8,hybrid,copy";
    int offset=0,count=2,tokens=128,rounds=1,prefill_mode=20;bool raw=false;
    for(int i=1;i<argc;i++){std::string a=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("missing argument");return std::string(argv[i]);};
      if(a=="--prompts")file=value();else if(a=="--methods")methods=value();else if(a=="--offset")offset=stoi(value());else if(a=="--count")count=stoi(value());else if(a=="--tokens")tokens=stoi(value());else if(a=="--rounds")rounds=stoi(value());else if(a=="--diagnostic-offset")diagnostic_offset=stoi(value());else if(a=="--prefill-mode")prefill_mode=stoi(value());else if(a=="--raw")raw=true;else throw std::runtime_error("unknown option "+a);
    }
    if(file.empty()||tokens<1||count<1)throw std::runtime_error("supply --prompts and positive counts");
    std::vector<std::string> prompts,arms;std::ifstream in(file);if(!in)throw std::runtime_error("cannot read prompts");std::string line;
    while(std::getline(in,line))if(!line.empty()&&line[0]!='#')prompts.push_back(line);
    std::stringstream ss(methods);while(std::getline(ss,line,',')){if(line!="serial"&&line!="df8"&&line!="df4"&&line!="df6"&&line!="adaptive"&&line!="copy"&&line!="hybrid"&&line!="recycle")throw std::runtime_error("unknown method");arms.push_back(line);}
    if(offset<0||offset>=(int)prompts.size()||rounds<1||(prefill_mode!=0&&prefill_mode!=20))throw std::runtime_error("invalid case range, rounds or prefill mode");
    Engine e;e.load(model,2,1,4096);e.prepare_batch(128,Engine::BATCH_WORKSPACE_ONLY);e.prepare_sequence();e.prefill_batch_mode=prefill_mode;e.load_dflash(drafter,2,Engine::DRAFT_Q4);
    Tokenizer tk;tk.load(model);std::cout<<std::setprecision(12);
    for(int round=0;round<rounds;round++)for(int ci=offset;ci<std::min((int)prompts.size(),offset+count);ci++){
        auto prompt=tk.encode(raw?prompts[ci]:Tokenizer::chat_prompt(prompts[ci],false),true);
        if(round%2)std::reverse(arms.begin(),arms.end());
        for(const auto &method:arms){auto r=run(e,tk,prompt,tokens,method);
            std::cout<<"{\"case\":"<<ci<<",\"round\":"<<round<<",\"method\":\""<<method<<"\",\"prompt_tokens\":"<<prompt.size()<<",\"prefill_s\":"<<r.prefill<<",\"decode_s\":"<<r.decode<<",\"request_s\":"<<r.prefill+r.decode<<",\"tokens_per_second\":"<<r.tokens.size()/r.decode<<",\"steps\":"<<r.steps<<",\"nodes\":"<<r.nodes<<",\"copy_steps\":"<<r.copy_steps<<",\"draft_steps\":"<<r.df_steps<<",\"recycle_steps\":"<<r.recycle_steps<<",\"draft_s\":"<<r.draft<<",\"verify_s\":"<<r.verify<<",\"next\":"<<r.next<<",\"tokens\":";array(r.tokens);std::cout<<",\"progress\":";array(r.progress);std::cout<<"}\n"<<std::flush;
        }
        if(round%2)std::reverse(arms.begin(),arms.end());
    }
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
