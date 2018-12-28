function ExecutionOrder_GM(obj, varargin)
%rerank executionOrder for the multi-branch module
opts.sectionLen = 2;
opts = vl_argparse(opts, varargin) ;

b5 = obj.getLayerIndex(['conv5_' num2str(opts.sectionLen) '_relu']);
if isnan(b5)
    error('please check the number of last layer before gating module')
end
obj.executionOrder(b5+1:end) = [b5+1 : numel(obj.executionOrder) ];


