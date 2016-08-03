classdef CNN < handle & AbstractNet
    % Implementation of AbstractNet for Convolutional Neural Networks
    % (requires MatConvNet backend).
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        nFilters;     % # of filters
        nChannels;    % # of channels
        fSz;          % filters dimensions
        stride;       % Stride
        inSz;         % input image size
        poolSz;       % pool size
        
        pretrainOpts; % pretraining options
        trainOpts;    % training options
        
        filters;      % filters weights
        b;            % biases
        
        gamma;
        beta;
        
        dWold;        % previous delta in weights
        dbold;        % previous delta in biases
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = CNN(inSz, filterSz, nFilters, pretrainOpts,trainOpts, varargin)
            % CNN Build a CNN instance
            %   obj = CNN([ih iw], [fh fw], N, O) returns an instance of
            %   CNN with N filters fh by fw large, accepting inputs of size
            %   ih by iw. O is a structure with the following fields:
            %     lRate     -- learning rate
            %     decayNorm -- weight decay norm, 1 for L1, 2 for L2 [optional]
            %     decayRate -- coefficient on weight decay penalty [optional]
            %     dropout   -- proportion of output coordinates zeroed out 
            %                  for regularization (a compensation factor is
            %                  applied to the remaining output) [optional]
            % 
            %   obj = CNN(inSz, filterSz, nFilters, trainOpts, 'option', value, 
            %   ...) generates a modified CNN according to the following 
            %   option/value pairs:
            %     'stride' -- [vs hs] vt and hz stride of the convolution
            %     'pool'   -- [pw ph] add a max-pooling layer with pw by ph 
            %                 pools
            %     'bias'   -- set to false to disable bias
            
            if numel(inSz) == 2
                inSz(3) = 1;
            end
            
            obj.nFilters  = nFilters;
            obj.nChannels = inSz(3);
            obj.stride    = [];
            obj.fSz       = filterSz;
            obj.inSz      = inSz;
            obj.poolSz    = [];
            if ~isfield(trainOpts, 'momentum')
                trainOpts.momentum = 0;
            end
            obj.trainOpts = trainOpts;

            obj.dWold = zeros(filterSz(1),filterSz(2),inSz(3),nFilters);
            obj.dbold = zeros(nFilters,1);
            obj.pretrainOpts = pretrainOpts;
            wRange        =sqrt(6/(inSz(3)*prod(filterSz)));
            obj.filters   = ...
                rand([filterSz inSz(3) nFilters], 'single') * wRange;
            obj.b         = zeros(nFilters, 1, 'single');
            
            assert(mod(numel(varargin), 2) == 0, ...
                'options should be ''option'', values pairs');
            for o = 1:2:numel(varargin)
                if strcmp(varargin{o}, 'bias') && ~varargin{o+1} % no bias
                    obj.b = [];
                elseif strcmp(varargin{o}, 'stride') % stride
                    assert(numel(varargin{o+1} == 2), ...
                        'stride should have two values');
                    obj.stride = reshape(varargin{o+1}, 1, 2);
                elseif strcmp(varargin{o}, 'pool')
                    assert(numel(varargin{o+1}) == 2, ...
                        'pool size should have two values');
                    obj.poolSz = reshape(varargin{o+1}, 1, 2);
                end
            end
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
        function S = insize(self)
            S = [self.inSz self.nChannels];
        end
        
        function S = outsize(self)
            if isempty(self.stride)
                S = self.inSz(1:2) - self.fSz + 1;
            else
                S = floor((self.inSz(1:2) - self.fSz) ./ self.stride) + 1;
            end
            if ~isempty(self.poolSz)
                S = floor(S ./ self.poolSz);
            end
            S = [S self.nFilters];
        end
        
        function [Y, A] = compute(self, X)
            assert(isa(X, 'single'), 'only single precision input supported');
            
            % Convolution
            X = reshape(X, self.inSz(1), self.inSz(2), self.nChannels, []);

            % Save values for backprop
            if nargin > 1
                A.X = X;
            end
            
            if ~isempty(self.stride)
                Y = vl_nnconv(X, self.filters, self.b, 'Stride', self.stride);
            else
                Y = vl_nnconv(X, self.filters, self.b);
            end
            
            % Dropout
            if nargout > 1 && isfield(self.trainOpts, 'dropout')
                rate = self.trainOpts.dropout;
                A.mask = rand([self.inSz(1:2) - self.fSz(1:2) + 1, ...
                               self.nFilters]) > rate;
                Y = bsxfun(@times, Y, single(A.mask / (1-rate)));
            end
            
            % Max-pooling
            if ~isempty(self.poolSz)
                if nargin > 1
                    A.Y = Y;
                end
                Y = vl_nnpool(Y, self.poolSz, 'Stride', self.poolSz, ...
                	'Method', 'max');
            end
            
            % Rectification
            %
            %Y = max(0, Y);
        end
        
        function [] = pretrain(self, X, preve)
            % Use an RBM to pretrain a filter
            patchMaker = PatchNet([self.inSz(1) self.inSz(2)],self.fSz,self.fSz-1);
            train = reshape(X,self.inSz(1),self.inSz(2),size(X,ndims(X))*self.nChannels);
                trainer = RBM(prod(self.fSz)*self.nChannels,self.nFilters,self.pretrainOpts, self.trainOpts);
                fulltrainX = cell(prod(self.inSz(1:2)-self.fSz + 1)*self.nChannels,1000);
                ex = randperm(size(X,ndims(X)),1000);
                for j = 1:1000
                    for i = 1:self.nChannels
                        fulltrainX(i:self.nChannels:end,j) = patchMaker.compute(train(:,:,ex(j)+i));
                    end
                end
                trainingX = reshape(cell2mat(fulltrainX),prod(self.fSz)*self.nChannels,prod(self.inSz(1:2)-self.fSz + 1)*1000);
                trainer.pretrain(trainingX,preve);
            self.filters = reshape(trainer.W,self.fSz(1),self.fSz(2),self.nChannels,self.nFilters);

        end
        
        function [G, inErr] = backprop(self, A, outErr, varargin)
            % Unpool and rectification derivation
            opts = self.trainOpts;
            if isfield(opts,'momentumChange') && opts.momentumChange == cell2mat(varargin{1})
                self.trainOpts.momentum = opts.momentumNew;
            end
            out = self.outsize();
            outErr = reshape(outErr,out(1),out(2),out(3),size(outErr,ndims(outErr)));
            if ~isempty(self.poolSz)
                outErr = vl_nnpool(A.Y, self.poolSz, outErr, ...
                    'Stride', self.poolSz);% .* (A.Y > 0);
            end
            % Dropout
            if isfield(self.trainOpts, 'dropout')
                outErr = bsxfun(@times, outErr, A.mask);
            end
            
            % Backprop
            if isfield(self.trainOpts, 'NAG')
                add = opts.lRate * self.dWold * opts.momentum;
            else
                add = zeros('like',self.dWold');
            end
                
            if ~isempty(self.stride)
                [inErr, G.dW, G.db] = vl_nnconv(A.X, self.filters+add, self.b, ...
                    outErr, 'Stride', self.stride);
            else
                [inErr, G.dW, G.db] = vl_nnconv(A.X, self.filters+add, self.b, ...
                    outErr);
            end
            G.dW = self.dWold * opts.momentum + (1-opts.momentum)*G.dW;
            G.db = self.dbold * opts.momentum + (1-opts.momentum)*G.db;
            self.dWold = G.dW;
            self.dbold = G.db;
        end
        
        function [] = gradientupdate(self, G)
            opts = self.trainOpts;
            % Gradient update
            self.filters = self.filters - opts.lRate * G.dW;
            if ~isempty(self.b)
                self.b = self.b - opts.lRate * G.db;
            end
            
            % Weight decay
            if isfield(opts, 'decayNorm') && opts.decayNorm == 2
                self.filters = self.filters - opts.decayRate * self.filters;
                self.b = self.b - opts.decayRate * self.b;
            elseif isfield(opts, 'decayNorm') && opts.decayNorm == 1
                self.filters = self.filters - opts.decayRate * sign(self.filters);
                self.b = self.b - opts.decayRate * sign(self.b);

            end
        end
    end
    
end
