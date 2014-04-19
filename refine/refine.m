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
%
% - Refine the ConvNets, whose last feature layer was used as the input for Neural Net
% - Generate several variants for model blending in the next step
%

%  	1. first load the features trained in the previous ConvNet step
%		- "data" is the matrix of train/valid/test features stacked
%	2. then refine the model with additional NNet as follows:


load('train_idx'); %train-idx is a pre-defined permutation of training data
load('label'); %outcomes in training data

devIds = 1+train_idx;


% reconstruction/prediction

% the id of the model being trained -- we train several models for blending later on
% see the description paper for more details
modelId = 1; 

trainPredicts_file	= ['trainPredicts' num2str(modelId) '.csv'];
validPredicts_file	= ['validPredicts' num2str(modelId) '.csv'];
testPredicts_file	= ['testPredicts' num2str(modelId) '.csv'];

rng(6789);

start = tic();


[dataSize,N] = size(data);

if 1
	disp('Prepare data');
	
	maxId 		= 61578; %original size of training data
	trainIds	= [1:60416]; %training ids used for model estimation, the rest is for validation
	
	validIds	= setdiff([1:maxId],trainIds);
	testIds		= setdiff([1:dataSize],[trainIds,validIds]);

	trainData	= single(data(trainIds,:));
	validData 	= single(data(validIds,:));
	testData 	= single(data(testIds,:));

	trainOutcomes	= label(trainIds,:);
	validOutcomes	= label(validIds,:);

	%data transformation, see the description paper for more details
	%trainData	= sqrt(trainData);
	%validData 	= sqrt(validData);
	%testData 	= sqrt(testData);
	
	[trainSize,N]	= size(trainData);
	[validSize,N]	= size(validData);
	[testSize,N]	= size(testData);

	%-- data normalisation ------------
	meanData1 	= mean(trainData);
	stdData1 	= std(trainData);

	meanData2 	= mean(testData);
	stdData2 	= std(testData);
	
	trainData	= (trainData - repmat(meanData1,trainSize,1)) ./ single(1e-5 + repmat(stdData1,trainSize,1));
	validData	= (validData - repmat(meanData2,validSize,1)) ./ single(1e-5 + repmat(stdData2,validSize,1));
	testData	= (testData - repmat(meanData2,testSize,1)) ./ single(1e-5 + repmat(stdData2,testSize,1));

	%adding noise to train data, see the description paper for more details
	%trainData		= [trainData + single(0.1*randn(size(trainData)))];
	[trainSize,N]	= size(trainData);

	disp(['Preparation time ' num2str(round(toc(start)))]);
end

taskSize = size(trainOutcomes,2); % taskSize = 37, which is the number of outcomes per image


validIds1	= [500:validSize];

validIds2 	= setdiff([1:validSize],validIds1);
validSize1	= length(validIds1);
validSize2 	= length(validIds2);

validOutcomes1	= validOutcomes(validIds1,:);
validOutcomes2	= validOutcomes(validIds2,:);


if 1 %refinement with Neural Net

	clear opt;
	
	opt.l2_penalty		= 1e-5;
	opt.report_interval	= 10;
	opt.nIters			= 1000; %number of iterations
	opt.HiddenSize		= 50; %number of hidden units    
	opt.nConts			= 1; 	%continuation steps
	opt.MaxNorm			= 0.5;	%max norm of weights coming to hidden units
	opt.norm_penalty	= 1;
	
	fprintf('\tRunning NNet, multitask mode...\n');

	outMeans	= mean(trainOutcomes);
	outStds 	= std(trainOutcomes);
	trainOutcomesX = (trainOutcomes  - repmat(outMeans,trainSize,1)) ./ repmat(outStds,trainSize,1);
	validOutcomesX = (validOutcomes  - repmat(outMeans,validSize,1)) ./ repmat(outStds,validSize,1);

	trainIds	= [1:trainSize]; featIds 	= [1:N];
	
	%trainIds	= [1:round(0.5*trainSize)]; %use only half of data
	%featIds 	= [1:round(0.5*N)]; %user only half of features

	D = 2000;
	opt.label	= [trainOutcomesX(trainIds(1:D),:); validOutcomesX(validIds1,:)];
	opt.LabelSize	= taskSize;
	nNetStruct	= nNet([trainData(trainIds(1:D),featIds);validData(validIds1,:)],[],opt,'train');

	trainPredicts	= repmat(outMeans,trainSize,1) + repmat(outStds,trainSize,1).*nNet(trainData(:,featIds),nNetStruct,opt,'test');
	validPredicts1	= repmat(outMeans,validSize1,1) + repmat(outStds,validSize1,1).*nNet(validData(validIds1,featIds),nNetStruct,opt,'test');
	validPredicts2	= repmat(outMeans,validSize2,1) + repmat(outStds,validSize2,1).*nNet(validData(validIds2,featIds),nNetStruct,opt,'test');
	testPredicts	= repmat(outMeans,testSize,1) + repmat(outStds,testSize,1).*nNet(testData(:,featIds),nNetStruct,opt,'test');
	validPredicts	= repmat(outMeans,validSize,1) + repmat(outStds,validSize,1).*nNet(validData(:,featIds),nNetStruct,opt,'test');
end


trainPredicts(trainPredicts < 0) = 0;
trainPredicts(trainPredicts > 1) = 1;

validPredicts1(validPredicts1 < 0) = 0;
validPredicts1(validPredicts1 > 1) = 1;

validPredicts2(validPredicts2 < 0) = 0;
validPredicts2(validPredicts2 > 1) = 1;

validPredicts(validPredicts < 0) = 0;
validPredicts(validPredicts > 1) = 1;

testPredicts(testPredicts < 0) = 0;
testPredicts(testPredicts > 1) = 1;

trainRMSE	= sqrt(sum(sum((trainOutcomes - trainPredicts).^2))/(trainSize*taskSize));
validRMSE1	= sqrt(sum(sum((validOutcomes1 - validPredicts1).^2))/(validSize1*taskSize));
validRMSE2	= sqrt(sum(sum((validOutcomes2 - validPredicts2).^2))/(validSize2*taskSize));

fprintf('RMSE, train: %.5f, valid1 %.5f, valid2 %.5f, time: %.1f\n',trainRMSE,validRMSE1,validRMSE2,toc(start));


%-- printing out the predictions -----
dlmwrite(['predicts/', trainPredicts_file],trainPredicts);
dlmwrite(['predicts/', validPredicts_file],validPredicts);
dlmwrite(['predicts/', testPredicts_file],testPredicts);


