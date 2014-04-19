function [ll,dl] = nNetGrad(param,opt)

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
% Neural network for regression with multiple outcomes. Computing the data log-likelihood and its gradient
%

epsilon = 1e-5; %smoothing of probabilities - avoiding the numerical problems

K = opt.K;
L = size(opt.label,2);
[dataSize,N] = size(opt.data);

%-- reshape parameters --
lastF = 0;
labelParam			= param(lastF+1:L); lastF = lastF + L;
labelHiddenParam	= reshape(param(lastF+1:lastF+L*K),L,K); lastF = lastF+L*K;
hiddenParam			= param(lastF+1:lastF+K); lastF = lastF+K;
hiddenVisibleParam	= reshape(param(lastF+1:lastF+K*N),K,N); lastF = lastF+K*N;

%--------

%forpropagation
hiddenVals = ones(dataSize,1)*hiddenParam + opt.data*hiddenVisibleParam';
hiddenActivations = 1 ./ (1 + exp(-hiddenVals));

labelVals = ones(dataSize,1)*labelParam + hiddenActivations*labelHiddenParam';

err		= opt.label - labelVals;
ll 		= -0.5*sum(sum(err.*err));

dGrad 	= err;

labelGrad	= mean(dGrad,1);
labelHiddenGrad	= dGrad'*hiddenActivations ./ dataSize;
propGrad = (dGrad*labelHiddenParam) .* hiddenActivations .* (1-hiddenActivations);

hiddenGrad			= mean(propGrad,1);
hiddenVisibleGrad	= propGrad'*opt.data ./ dataSize;


%--------------------- parameter regularisation ------------------
param1 = param;
param1(1:L*(K+1)+K) = 0;

epsilon = 1e-8;
norm1 = sqrt(epsilon + param1.*param1);

ll = ll/dataSize - 0.5*opt.l2_penalty*param*param' - opt.l1_penalty*sum(norm1);

%controlling the norm
if opt.norm_penalty
	hiddenNorms = sqrt(sum(hiddenVisibleParam.^2,2));

	ids 	= (hiddenNorms >= opt.MaxNorm);
	ids2	= repmat(ids,1,N);

	diffNorms = hiddenNorms - opt.MaxNorm;
	ll = ll - 0.5*opt.norm_penalty*sum(diffNorms(ids).^2);

	normGrad = hiddenVisibleParam.*repmat((diffNorms./hiddenNorms),1,N);
	hiddenVisibleGrad(ids2) = hiddenVisibleGrad(ids2) - opt.norm_penalty*normGrad(ids2);
end	


grad = [labelGrad,reshape(labelHiddenGrad,1,L*K),hiddenGrad,reshape(hiddenVisibleGrad,1,K*N)];
dl = grad - opt.l2_penalty*param - opt.l1_penalty*param1./norm1;

