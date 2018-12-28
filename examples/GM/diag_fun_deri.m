% function  out_S = diag_fun_deri(S, epsilon, type, par) %lph 2016/11/11 5:50
function  out_S = diag_fun_deri(S, opts, wrt_which)

        switch opts.regu_method

            case {'power', 'power+1st_order', 'AF', 'power+mat-fro'}
                if isfield(opts, 'alpha')
                    out_S = opts.alpha .* ((S + opts.epsilon) .^ (opts.alpha - 1));
                else
                    out_S = 0.5 .* ( 1 ./ sqrt(S + opts.epsilon) );
                end
                
            case {'power+mat-l2', 'mat-l2'}  % cannot be here
                  error('Should not be here !');
                  
            case {'log'}
                % out_S = 1 ./ (S + epsilon);
                out_S = 1 ./ (S + opts.epsilon);
                
            case {'Burg'}
                k = length(S);
                beta = (1 - opts.alpha) / (2 * opts.alpha);
                P = exp(opts.weights{1}(1:k));
                if strcmp(wrt_which, 'wrt_S')
                    out_S = 0.5 * P ./ sqrt(beta^2 .*  P .^ 2 + P .* S + opts.epsilon);
                else
                    out_S = 0.5 * (2 * beta^2 .*  P .^ 2 + P .* S) ./ sqrt(beta^2 .*  P .^ 2 + P .* S + opts.epsilon) - beta .* P;
                end
                
            otherwise
                    error('derivative of diagonal matrix function not supported!');
        end
end
