classdef SoftmaxPerceptron < handle & AbstractNet
    % PERCEPTRON Single Layer Perceptron
    %   PERCEPTRON implements AbstractNet for single layer perceptrons of 
    %   neurons with bias. Neurons use sigmoïd activation.
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        W;         % connection weights
        b;         % bias
        trainOpts; % training options
        
        dWold;
        dbold;
        
        gamma;
        beta;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = SoftmaxPerceptron(inSz, outSz, trainOpts)
            % PERCEPTRON Constructor for RELURBM
            %   net = PERCEPTRON(inSz, outSz, O) returns an instance
            %   PERCEPTRON with inSz input neurons fully connected to outSz
            %   output neurons. Training setting are stores in the
            %   structure O with the fields:
            %       lRate     -- learning rate
            %       dropout   -- input units dropout rate [optional]
            %       decayNorm -- type of weight decay penalty [optional]
            %       decayRate -- coeeficient on weight decay penalty
            %   By default, connection weights are initialized using a
            %   centered Gaussian distribution of variance 1/inSz.
            
            if isfield(trainOpts, 'decayNorm') ...
                    || isfield(trainOpts, 'decayRate')
                assert(isfield(trainOpts, 'decayNorm') ...
                    && isfield(trainOpts, 'decayRate'), ...
                    'specify both decay norm and rate');
            end
            
            if ~isfield(trainOpts, 'momentum')
                trainOpts.momentum = 0;
            end
            
            obj.trainOpts = trainOpts;

            % Initializing weights
            range = sqrt(6/(outSz*inSz));
            obj.W = 2*range * (rand(inSz, outSz) - .5);
            obj.dWold = zeros(inSz,outSz);
            obj.b = zeros(outSz, 1);
                        obj.gamma =  2*range*(rand(inSz,1)-.5);
                        obj.beta = zeros(inSz,1);
            obj.dbold = obj.b;
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
        function S = insize(self)
            S = size(self.W, 1);
        end
        
        function S = outsize(self)
            S = size(self.W, 2);
        end
        
        function [Y, A] = compute(self, X)
            % training with dropout
            if isfield(self.trainOpts, 'DAG')
                add = opts.lRate * self.dWold * opts.momentum;
            else
                add = 0;
            end
            if isfield(self.trainOpts,'normInput')
                [X,A.Xmu,A.Var] = whiten(X,0.0001);
                X = diag(self.gamma)*X+repmat(self.beta,1,size(X,ndims(X)));
            end
            if nargout == 2 && isfield(self.trainOpts, 'dropout')
                A.mask  = rand(self.insize(), 1) > self.trainOpts.dropout;
                Wmasked = bsxfun(@times, self.W, ...
                    A.mask ./ (1 - self.trainOpts.dropout));
                % Save necessary values for gradient computation
                A.S = bsxfun(@plus, Wmasked' * X + add, self.b); % stimuli
                A.X = X;
                Y   = self.activation(A.S);
                A.Y = Y;
            elseif nargout == 2 % training
                % Save necessary values for gradient computation
                
                A.S = bsxfun(@plus, self.W' * X+add, self.b); % stimuli
                A.X = X;
                Y   = self.activation(A.S);
                A.Y = Y;
            else % normal
                Y = self.activation(bsxfun(@plus, self.W' * X, self.b));
            end
        end
        
        function [] = pretrain(~, ~)
            % Nothing to do
        end
        
        function [G, inErr] = backprop(self, A, outErr, varargin)            
            % Gradient computation
            %delta  = outErr .* A.Y .* (1 - A.Y);
            % Cross entropy wise :
            if isfield(self.trainOpts,'momentumChange') && self.trainOpts.momentumChange == cell2mat(varargin{1})
                self.trainOpts.momentum = self.trainOpts.momentumNew;
            end
            delta = A.Y - outErr;
            G.dW   = A.X * delta';
            G.db   = sum(delta, 2);
            
            G.dW = self.dWold * self.trainOpts.momentum + (1-self.trainOpts.momentum)*G.dW;
            G.db = self.dbold * self.trainOpts.momentum + (1-self.trainOpts.momentum)*G.db;
            self.dWold = G.dW;
            self.dbold = G.db;
            % Error backpropagation
            inErr = self.W * delta;
            if isfield(self.trainOpts,'normInput')
                dX = inErr .* repmat(self.gamma,1,size(inErr,2));
                dsigma = -1/2 * sum(dX .* A.Xmu .* repmat(A.Var .^(-3/2),1,size(inErr,2)),2);
                dmu = -1 * sum(dX .* repmat(A.Var.^(-1/2),1,size(inErr,2)),2);
                dmu = dmu + dsigma .* sum(-2*A.Xmu,2)/size(inErr,2);
                inErr = dX .* repmat(A.Var .^ (-1/2),1,size(inErr,2)) + repmat(dsigma,1,size(inErr,2)) * 2 .* A.Xmu / size(inErr,2) + repmat(dmu,1,size(inErr,2)) / size(inErr,2);
                G.dgamma = sum(A.X .* repmat(self.gamma,1,size(inErr,2)),2);
                G.dbeta = sum(inErr,2);
            end
            
            % Dropout
            if isfield(self.trainOpts, 'dropout')
                G.dW  = bsxfun(@times, G.dW, A.mask) ...
                    * (1 - self.trainOpts.dropout);
                inErr = bsxfun(@times, inErr, A.mask) ...
                    * (1 - self.trainOpts.dropout);
            end
        end
        
        function [] = gradientupdate(self, G)
            opts = self.trainOpts;
            % Gradient update
            self.W = self.W - opts.lRate * G.dW;
            self.b = self.b - opts.lRate * G.db;
            if isfield(opts,'normInput')
                self.gamma = self.gamma - opts.lRate * G.dgamma;
                self.beta = self.beta - opts.lRate*G.dbeta;
            end
            
            % Weight decay
            if isfield(opts, 'decayNorm') && opts.decayNorm == 2
                self.W = self.W - opts.decayRate * self.W;
                self.b = self.b - opts.decayRate * self.b;
            elseif isfield(opts, 'decayNorm') && opts.decayNorm == 1
                self.W = self.W - opts.decayRate * sign(self.W);
                self.b = self.b - opts.decayRate * sign(self.b);
            elseif isfield(opts, 'decayNorm') && opts.decayNorm == 3
                mat = min(abs(self.W),sqrt(opts.decayRate));
                self.W = sign(self.W) .* mat;
            end
        end
        
    end % methods
    
    methods(Static)
        
        function [Y] = activation2(X)
            % Sigmoïd activation
            Y = 1 ./ (1 + exp(-X));
        end
        function [Y] = activation(X)
            % softmax activation
            Y = softmax(X);
        end
    end
    
end % classdef RBM
