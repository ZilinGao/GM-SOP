classdef MPN_COV_Pool < dagnn.Layer
  properties
    method = 'covariance_small'
    regu_method = 'power'
    alpha = 0.5
    epsilon = 1e-12
    opts = {'cuDNN'}
  end

  methods
      
    %% forward  function
    function outputs = forward(self, inputs, params)
        % function upper = nn_forward_cov_pool(opts, lower, upper, masks)
        % res(i+1) = l.forward(l, res(i), res(i+1)) ;
        [outputs{1} outputs{2} outputs{3}] = nn_forward_covariance_EIG_single(self, inputs{1});
    end

    % V1-2(checked, untested) 20161213, modified for class 2017/02/05
    function [upper_x_  V  S]= nn_forward_covariance_EIG_single(opts, lower_x_)  % EIG-based: verison of SINGLE precision 2016.10.17 9:00  2016.11.18

            [M,N,D,L] = size(lower_x_);

            gpuMode = isa(lower_x_, 'gpuArray') ;  
            if gpuMode
                lower_x = (gather(lower_x_));
                if isfield(opts, 'weights') 
                    if ~isempty(opts.weights)
                        opts.weights{1} = gather(opts.weights{1});
                    end
                end
            else
                lower_x = (lower_x_);
            end

            n = M * N;

            X = reshape(lower_x, [n, D, L]);

            if ~strcmp(opts.method, 'covariance_small')
                upper_x = zeros(1, 1, D * D, L, 'single');      
            else
                A = ones(D, D, 'single'); A=triu(A); ids=(A>0);   
                upper_x = zeros(1, 1, D * (D+1) / 2, L, 'single');    
            end

            % alocate matrices
            S = zeros(D, D, L, 'single'); % eigenvalue matrix
            V = zeros(D, D, L, 'single'); 
%             opts.epsilon = 0.1 ;
            for i = 1: L
                
               % [upper_x  V  S]   = nn_foward_one_image_covariance(X, opts, ids)
                [upper_x(1,1, :, i)  V(:, :, i)  S(:, :, i)] = nn_foward_one_image_covariance(X(:, :, i),  opts,  ids);

            end

            if gpuMode
                upper_x_ = gpuArray((upper_x));
            else
                upper_x_ = (upper_x);
            end
            % upper_x_aux{1} = V;
            % upper_x_aux{2} = S;

    end

    function [upper_x  V  S] = nn_foward_one_image_covariance(X, opts, ids)

               n = size(X, 1);

               P       = cov(X,  1);
               [V, S] = eig(P);

               diag_S = diag(S) ;
               [diag_S, idx] = sort(diag_S, 'descend');
               V = V(:, idx);

               % ind =diag_S  > (D * eps(max(diag_S))); % ! changed 
               ind    = diag_S  > ( eps(max(diag_S))); 
               Dmin = min(find(ind, 1, 'last'), n);

               V(:, Dmin + 1 : end) = 0; 
               S(:) = 0;
               S(1 : Dmin, 1 : Dmin) = diag(diag_S(1:Dmin));

               % lph 2016/11/11 5:50
               % P = V(:, 1:Dmin, i) * diag(diag_fun(diag_S(1:Dmin), opts.epsilon, 'power', 0.5)) * V(: , 1:Dmin, i)';
               P = V(:, 1:Dmin) * diag(diag_fun(diag_S(1:Dmin), opts)) * V(: , 1:Dmin)';

               if ~strcmp(opts.method, 'covariance_small')
                   upper_x = P(:);
               else
                   upper_x = P(ids);
               end

    end

     %% backwardAdvanced is modified slightly for custom layer
    function backwardAdvanced(obj, layer)
    %BACKWARDADVANCED Advanced driver for backward computation
    %  BACKWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
    %  the backward step of the layer.
    %
    %  The advanced interface can be changed in order to extend DagNN
    %  non-trivially, or to optimise certain blocks.
      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      par = layer.paramIndexes ;
      net = obj.net ;

      inputs = {net.vars(in).value} ;
      derOutputs = {net.vars(out).der} ;  
      % for i = 1:numel(derOutputs)  % lph 2017/02/5 17:12
      %   if isempty(derOutputs{i}), return ; end
      % end
      outputs = {net.vars(out).value};  
      if isempty(derOutputs{1}), return; end

      if net.conserveMemory
        % clear output variables (value and derivative)
        % unless precious
        for i = out
          if net.vars(i).precious, continue ; end
          net.vars(i).der = [] ;
          net.vars(i).value = [] ;
        end
      end

      % compute derivatives of inputs and paramerters
      % lph 2017/02/05 10:37
      % [derInputs, derParams] = obj.backward(inputs, {net.params(par).value}, derOutputs) ;
      [derInputs, derParams] = obj.backward(inputs, {net.params(par).value}, derOutputs, outputs) ;
      if ~iscell(derInputs) || numel(derInputs) ~= numel(in)
        error('Invalid derivatives returned by layer "%s".', layer.name);
      end

      % accumuate derivatives
      for i = 1:numel(in)
        v = in(i) ;
        if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
          net.vars(v).der = derInputs{i} ;
        elseif ~isempty(derInputs{i})
          net.vars(v).der = net.vars(v).der + derInputs{i} ;
        end
        net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1 ;
      end

      for i = 1:numel(par)
        p = par(i) ;
        if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
              || isempty(net.params(p).der)
          net.params(p).der = derParams{i} ;
        else
          net.params(p).der = vl_taccum(...
            1, net.params(p).der, ...
            1, derParams{i}) ;
          %net.params(p).der = net.params(p).der + derParams{i} ;
        end
        net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1 ;
        if net.numPendingParamRefs(p) == net.params(p).fanout
          if ~isempty(net.parameterServer) && ~net.holdOn
            net.parameterServer.pushWithIndex(p, net.params(p).der) ;
            net.params(p).der = [] ;
          end
        end
      end
    end
   
    %% backward function
    function [derInputs, derParams] = backward(self, inputs, params, derOutputs, outputs)
        
       derInputs{1} = nn_backward_covariance_EIG_single_weights_free(self, inputs{1}, derOutputs{1}, outputs);
       derParams  = {} ;
       
    end
 
    
    % V1-2(checked, untested) 20161213
    function lower_dzdx_ = nn_backward_covariance_EIG_single_weights_free(opts, lower_x_, upper_dzdx_, upper_x_)   %EIG-based: verison of SINGLE precision 2016.10.16 10:00
          upper.aux{1} = upper_x_{2};
          upper.aux{2} = upper_x_{3};
               
          [M, N, D, L] = size(lower_x_);

          n = M * N;

          gpuMode = isa(lower_x_, 'gpuArray') ;  
          if gpuMode 
               upper_dzdx = (gather(upper_dzdx_));
               lower_x = (gather(lower_x_));
          else
              upper_dzdx = (upper_dzdx_);
              lower_x = (lower_x_);
          end
          X = reshape(lower_x, [n, D, L]); 

          lower_dzdx = zeros(M, N, D, L, 'single');

          if ~strcmp(opts.method, 'covariance_small')
          else
              A = ones(D, D, 'single'); A=triu(A); ids=(A>0);   
          end

          I_ = (2 / n) .* (eye(n, n, 'single') - (1 / n) .* ones(n, 1, 'single') * ones(1,n, 'single'));
          for i = 1 : L  % iterate over images in batch

               % lower_dzdx = nn_backward_one_image(I_, V, S, upper_dzdx, M, N);
               lower_dzdx(:,:,:,i) = nn_backward_one_image_covariance(I_,  X(:, :, i), upper.aux{1}(:, :, i), upper.aux{2}(:, :, i), upper_dzdx(1, 1,  :,  i), ids, opts, M, N); % faster 2016/12/11 14:43

          end

          if gpuMode
                lower_dzdx_ = gpuArray((lower_dzdx));
          else
                lower_dzdx_ = (lower_dzdx);
          end

    end


    function   lower_dzdx = nn_backward_one_image_covariance(I_,  X,  V,  S,  upper_dzdx,  ids,   opts, M, N)

                D = size(V, 1);
                n = M * N;

                dLdC = zeros(D,  D,  'single');
                 if ~strcmp(opts.method, 'covariance_small')
                     dLdC(:) = upper_dzdx(:);
                 else
                     dLdC(ids) = upper_dzdx(:);

                      dLdC = (dLdC + dLdC') / 2;  % 2017/01/25 6:12
                 end

                diag_S = diag(S);

                % ind =diag_S  > (D * eps(max(diag_S))); % ! changed 
                ind =diag_S  > ( eps(max(diag_S))); 
                Dmin = min(find(ind, 1, 'last'), n);


                dLdV = 2 *  symmetric(dLdC) * V(:, 1:Dmin) * diag(diag_fun(diag_S(1:Dmin), opts));   
                if any(strcmp(opts.regu_method, {'power+mat-l2', 'mat-l2'}))      % lph 2017/01/21 13:55
                     diag_S_alpha = diag_S(1:Dmin) .^ opts.alpha;  
                     diag_S_alpha = diag_S_alpha / diag_S_alpha(1) * (opts.alpha / diag_S(1));
                     % dLdS = diag_S(1) * diag_S_alpha ./ diag_S(1:Dmin)  .* diag( V(:, 1:Dmin)' * dLdC * V(:, 1:Dmin) );  % alternative method
                     % dLdS(1) = dLdS(1) - trace(dLdC * V(:, 1:Dmin) * diag(diag_S_alpha(1:Dmin)) * V(:, 1:Dmin)');
                     z = diag(V(:, 1:Dmin)' * dLdC * V(:, 1:Dmin));
                     dLdS = diag_S(1) * diag_S_alpha ./ diag_S(1:Dmin) .* z;    dLdS(1) = dLdS(1) - sum( diag_S_alpha .* z);
                     dLdS = diag(dLdS);

                else
                      dLdS =   diag(diag_fun_deri(diag_S(1:Dmin), opts)) * ( V(:, 1:Dmin)' * dLdC * V(:, 1:Dmin) );
                end


                %temp_K = diag_S(1:Dmin) .^ 2 * ones(1, Dmin);  %lph
                K(1:Dmin, 1:Dmin) = diag_S(1:Dmin)  * ones(1, Dmin, 'single');
                K(1:Dmin, 1:Dmin)  = 1./ (K(1:Dmin, 1:Dmin)  - K(1:Dmin, 1:Dmin)');
                %K(eye(size(K,1))>0)=0;K
                
                K(isinf(K)) = 0;  
  
                %dzdx =   U * ( dDiag(dLdS(:, 1:Dmin))  +   2  * S(:, 1:Dmin) * symmetric(K' .* (V(:, 1:Dmin)' * dLdV(:, 1:Dmin))) ) * V(:, 1:Dmin)';
                 dzdx =   I_ * X * ...
                              symmetric( V(:, 1:Dmin) * (diag(diag( dLdS ))  +  K(1:Dmin, 1:Dmin)' .* (V(:, 1:Dmin)' * dLdV) ) * V(:, 1:Dmin)' );

                 lower_dzdx = reshape(dzdx,  [M N D]); 

    end



    %% constructor
    function obj = MPN_COV_Pool(varargin)
      obj.load(varargin) ;
    end
  end
end
