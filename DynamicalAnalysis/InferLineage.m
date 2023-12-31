function out = InferLineage(DA_results, par)
% Infer the cell lineage by MPPT or MPFT approach based on MuTrans results
% Input :
%   DA_results: The output structure generated by DynamicalAnalysis function
%   par.root (optional for MPFT): the root node of cell lineage
%   par.method: different approaches to infer cell lineage. Can be either
%   'mppt' or 'mpft';
%   the following parameters only applied in MPPT method:
%   par.trim: the logical value, whether to trim the rwTPM before MPPT
%   inference. Default is false (might cause some short-cuts).
%   par.norm: the logical value, whether to exclude the self-transition
%   probability and renormalize the rwTPM. Default is false.
%   par.thresh_prob: the threshold of minimum probability to keep in trimmed
%   rwTPM. Default is 0.
%   par.thresh_sharp: the threshold of maximum transition sharpness \theta
%   (by fitting logistic function of TCS) to keep in rwTPM. Default is 100.
%   par.otherkeep: the threshold of minimum membership function included in
%   estimating sharpness \theta. Default is 0.1.
%   
if ~isfield(par,'method')
par.method = 'mpft';
end

if ~isfield(par,'thresh_prob')
par.thresh_prob = 0;
end

if ~isfield(par,'thresh_sharp')
par.thresh_sharp = 100;
end

if ~isfield(par,'otherkeep')
par.otherkeep = 0.1;
end


%%
P_hat = DA_results.P_hat;
rho_class = DA_results.rho_class;
class_order = DA_results.class_order;
mu_hat = DA_results.mu_hat;
K = max(class_order);

if strcmp(par.method,'mppt')

    k = size(P_hat,1);

    if ~isfield(par,'trim')
    par.trim = false;
    end

    if ~isfield(par,'norm')
    par.norm = false;
    end

    if par.norm
    P_hat = (eye(k)-diag(diag(P_hat)))^(-1)*P_hat;
    P_hat= P_hat -diag(diag(P_hat));
    end

    figure;hist(P_hat)

    if par.trim
       [P_hat,theta_matrix] = TPM_trim (P_hat,class_order, rho_class, par);
       out.theta = theta_matrix;
    end

    keep_id = P_hat > par.thresh_prob;
    mod_id = ~keep_id;
    P_hat_mod = P_hat;
    P_hat_mod(mod_id) = 0;

    Prob_weight = -log(P_hat_mod);
    G_cg = digraph(Prob_weight);
    figure;
    TR = shortestpathtree(G_cg,par.root);
    plot(TR);
    out.P_hat_mod = P_hat_mod;
    out.tree = TR;
else
    if strcmp(par.method,'mpft')
        S = -diag(mu_hat)*P_hat;% the prob. flow matrix

        G_s = graph(S,'upper');
        
        if ~isfield(par,'root')
                [~,pred] = minspantree(G_s);
        else
            [~,pred] = minspantree(G_s,'root',par.root);
        end
        
        rootedTree = digraph(pred(pred~=0),find(pred~=0));
        
        
        if ~isfield(par,'legend_text')
            par.legend_text = 1:K;
        end
        
        if ~isfield(par,'Nodesize')
            par.Nodesize = 40;
        end
        
        if ~isfield(par,'Lwidth')
            par.Lwidth = 4.0;
        end

        if ~isfield(par,'colors')
            par.colors = brewermap(K,'set1');
        end
        
        if ~isfield(par,'NodeFontSize')
            par.NodeFontSize = 20;
        end
        
        figure;
        plot(rootedTree,'Marker','o','MarkerSize',par.Nodesize,'NodeColor',par.colors(1:K,:),'LineWidth',par.Lwidth,'NodeLabel',par.legend_text,'NodeFontSize',20,'EdgeColor',[0.69 0.77 0.87]);
        axis off;
    end
end
end