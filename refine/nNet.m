function [output] = nNet(data,param0,opt,mode)

% Copyright (c) 2014, Truyen Tran (tranthetruyen@gmail.com)
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification,
% are permitted provided that the following conditions are met:
%
% - Redistributions of source code must retain the above copyright notice,
%   this list of conditions and the following disclaimer.
%
% - Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
% EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

%
% Neural network for regression with multiple outcomes
%

% Usages:
%	For training:	[nNetStruct]	= nNet(train_data,param0,opt,'train')
%	For testing:	[output] 		= nNet(test_data,nNetStruct,opt2,'test');
%
% Fields:
%	param0						: initial parameters
%		param0.Label			: label bias
%		param0.LabelHidden		: mapping from hidden units to labels
%		param0.Hidden			: hidden unit bias
%		param0.HiddenVisible	: mapping from visible inputs to hidden units
%
%	opt [mode = training]
%		opt.label				: training label, a vector or a matrix of 0/1 (for multiclasses), opt.label(n,l)=1 if the example 'n' has label 'l', and 0 otherwise
%		opt.epsilon				: threshold -- training stops of relative improvement over log-likelihood falls below this threshold. Typically choose [1e-4, 1e-7]. Default: 1e-5
%		opt.nIters				: maximum number of tranining iterations. Default: 100
%		opt.report_interval		: time between reports of learning progress. Default: 1s
%		opt.l2_penalty			: quadratic penalty of parameters. Default: 1e-4
%
%	output [mode = training]	: the model parameters
%		output.Label
%		output.LabelHidden
%		output.Hidden
%		output.HiddenVisible
%
%	output  [mode = testing]
%		output 			: 

start = tic();
last_time = toc(start);

[dataSize,N] = size(data);

if ~isempty(param0)
	K = length(param0.Hidden);
	L =	length(param0.Label);
else
	
	if ~isfield(opt,'HiddenSize')
		disp('Missing hidden size');
		return;
	end
	
	if ~isfield(opt,'LabelSize')
		disp('Missing label size');
		return;
	end
	
	K = opt.HiddenSize;
	L = opt.LabelSize;
end

if strcmp(mode,'train') | strcmp(mode,'training') | strcmp(mode,'learn') | strcmp(mode,'learning')

	%-- hyper-parameter setting ----
	epsilon = 1e-5;
	if isfield(opt,'epsilon');
		epsilon = opt.epsilon;
	end
	
	if ~isfield(opt,'nIters')
		opt.nIters = 100;
	end
	
	if ~isfield(opt,'report_interval')
		opt.report_interval = 1;
	end

	if ~isfield(opt,'l1_penalty')
		opt.l1_penalty = 0;
	end
	
	if ~isfield(opt,'l2_penalty')
		opt.l2_penalty = 1e-4;
	end
	
	if ~isfield(opt,'norm_penalty')
		opt.norm_penalty = 0;
	end

	if ~isfield(opt,'MaxNorm')
		opt.MaxNorm = 100;
	end

	isSmallBatch = 1;
	if isfield(opt,'isSmallBatch')
		isSmallBatch = opt.isSmallBatch;
	end
	
	nConts = 50;
	if isfield(opt,'nConts')
		nConts = opt.nConts;
	end
	
	%-------	

	%param initialization
	if isempty(param0)
		param0.Label 			= zeros(1,L);
		param0.LabelHidden		= 0.01*randn(L,K);
		param0.Hidden			= zeros(1,K);
		param0.HiddenVisible	= 0.01*randn(K,N);
		param0.LabelVisible		= zeros(1,L*N);
	end

	%------- reshape parameters -------------
	param = [];
	param = [param,param0.Label,reshape(param0.LabelHidden,1,L*K),param0.Hidden,reshape(param0.HiddenVisible,1,K*N)];
	paramSize	= length(param);
	%------- reshape parameters -------------

	opt.K				= K;
	opt.data 			= data;

	%--- training small batches ------------
	if isSmallBatch
		opt2 = opt;
		for iter=1:10
			if dataSize < 5000
				blockSize = dataSize;
			elseif dataSize < 10000
				blockSize = 3000;
			else
				blockSize = 5000;
			end

			randIds = randperm(dataSize);

			p = 0;
			for d=1:blockSize:dataSize
				p = p + 1;
				dMax = min(d+blockSize-1,dataSize);

				opt2.data		= data(randIds(d:dMax),:);
				opt2.label		= opt.label(randIds(d:dMax),:);
				opt2.l2_penalty	= min(0.1,opt.l2_penalty*min(100,dataSize/blockSize));
				opt2.epsilon	= 1e-5;

				[param,fvals,it] = conjugate_gradient('nNetGrad',param,opt2,opt2.epsilon,round(0.5*opt.nIters/10),opt.report_interval);

				curr_time = toc(start);
				if curr_time >= last_time + opt.report_interval
					fprintf('Small patch %d, iter: %d, ll: %.5f, time: %.1f\n',p,iter,fvals(it),curr_time);
					last_time = curr_time;
				end
			end
		end

		%--- refinement -------
		[param,fvals,it] = conjugate_gradient('nNetGrad',param,opt,opt.epsilon,round(0.5*opt.nIters),opt.report_interval);
	else
		[param,fvals,it] = conjugate_gradient('nNetGrad',param,opt,opt.epsilon,opt.nIters,opt.report_interval);
	end

	curr_time = toc(start);
	fprintf('Local search, ll: %.5f, time: %.1f\n',fvals(it),curr_time);


	%---- continuation method ---------
	if nConts
		fprintf('Starting continuation method search, time: %.1f\n',curr_time);

		bestParam	= param;
		bestFval 	= fvals(it);

		l2_penalty0 	= opt.l2_penalty;
		norm_penalty0	= opt.norm_penalty;

		nFails = 0;
		last_time = curr_time;
		for iter=1:nConts

			scale = 200*rand;

			%energy deformation				
			opt.l2_penalty 		= min(0.01,scale*l2_penalty0);
			opt.norm_penalty	= scale*norm_penalty0;

			[param,fvals,it] = conjugate_gradient('nNetGrad',param,opt,opt.epsilon,opt.nIters,opt.report_interval);

			%local seeking
			opt.l2_penalty 		= l2_penalty0;
			opt.norm_penalty	= norm_penalty0;

			[param,fvals,it] = conjugate_gradient('nNetGrad',param,opt,opt.epsilon,opt.nIters,opt.report_interval);

			if bestFval >= fvals(it)
				nFails = nFails + 1;

				if nFails >= 5
					break;
				end
			else
				nFails 		= 0;
				bestFval 	= fvals(it);
				bestParam	= param;
			end

			curr_time = toc(start);
			if curr_time >= last_time + opt.report_interval
				fprintf('Continuation method, iter %d, ll: %.5f, time: %.1f\n',iter,bestFval,curr_time);
				last_time = curr_time;
			end

		end
		fprintf('Global search, ll: %.5f, time: %.1f\n',bestFval,toc(start));

		param = bestParam;
	end
	
	%----- reshape parameters ---------
	lastF =0;
	output.Label			= param(lastF+1:lastF+L); lastF = lastF + L;
	output.LabelHidden		= reshape(param(lastF+1:lastF+L*K),L,K); lastF = lastF+L*K;
	output.Hidden			= param(lastF+1:lastF+K); lastF = lastF+K;
	output.HiddenVisible	= reshape(param(lastF+1:lastF+K*N),K,N); lastF = lastF+K*N;
	%----- END reshape parameters ---------

	
elseif strcmp(mode,'test') | strcmp(mode,'testing') | strcmp(mode,'predict') | strcmp(mode,'predicting') | strcmp(mode,'prediction')
	
	hiddenVals = ones(dataSize,1)*param0.Hidden +  data*param0.HiddenVisible';
	hiddenActivations = 1 ./ (1 + exp(-hiddenVals));

	output = ones(dataSize,1)*param0.Label + hiddenActivations*param0.LabelHidden';
else
	fprintf('Err: Dont know the mode %s!!\n',mode);
end
