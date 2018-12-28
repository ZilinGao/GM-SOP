% function  out_S = diag_fun(S, epsilon, type, par) %lph 2016/11/11 5:50
function  out_S = diag_fun(S, opts)

        switch opts.regu_method
            case {'power', 'power+1st_order', 'AF', 'power+mat-fro'}
                if isfield(opts, 'alpha')
                      out_S =  (S  + opts.epsilon) .^ opts.alpha;
                else
                      out_S =  sqrt(S  + opts.epsilon);
                end

            case {'power+mat-l2', 'mat-l2'}
                out_S =  (S  + opts.epsilon) .^ opts.alpha;
                out_S =  out_S / out_S(1);
                
            case {'log'}
                % out_S = log(S+epsilon);
                out_S = log(S + opts.epsilon);
                
            case {'Burg'}
                k = length(S);
                beta = (1 - opts.alpha) / (2 * opts.alpha);
                P = exp(opts.weights{1}(1:k));
                out_S = sqrt(beta ^ 2 .* P .^ 2 + P .* S + opts.epsilon) - beta .* P;
                
            otherwise
                    error('diagonal matrix function not supported!');
        end
end