classdef RBM < handle & AbstractNet
    % RBM Restricted Boltzmann Machine Model
    %   RBM implements AbstractNet for Restricted Boltzmann Machines with 
    %   binary units on both visible and hidden units.
    %
    %   Pretraining regularization includes L2 and L1 weight decay, dropout and 
    %   hidden units sparsity.

    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        nVis;                  % # of visible units (dimensions)
        nHid;                  % # of hidden units
        W;                     % connection weights
        b;                     % visible unit biases
        c;                     % hidden unit biases
        beta;                  % 
        
        pretrainOpts = struct();
        trainOpts    = struct();
    end % properties
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = RBM(nVis, nHid, pretrainOpts, trainOpts)
            % RBM Construct a Restrcted Boltzmann Machine implementation
            %   obj = RBM(nVis, nHid, pretrainOpts, trainOpts) returns an
            %   RBM with nVis binary input units, nHid binary output units.
            %   The structure pretrainOpts contains pretraining parameters
            %   in the following fields:
            %       lRate             -- learning rate
            %       batchSz           -- # of samples per batch
            %       momentum          -- gradient momentum [optional]
            %       momentumChange    -- change in gradient momentum
            %                         [optional]
            %       momentumChangeDif -- when the change in gradient
            %                         momentum happens. This is dependant
            %                         on reconstruction error being calculated
            %                         so it needs displayEvery [optional]
            %       sampleVis         -- sample visible units for GS (default:
            %                         false)
            %       sampleHid         -- sample hidden units for GS (default:
            %                         true)
            %       nGS               -- # GS iterations for CD(k)
            %       wPenalty          -- weight penalty [optional]
            %       wDecayDelay       -- first epoch with weight decay [optional]
            %       dropHid           -- hidden units dropout rate [optional]
            %       dropVis           -- hidden units dropout rate [optional]
            %       sparsity          -- sparseness objective [optional]
            %       sparseGain        -- learning rate gain for sparsity [optional]
            %       decayNorm         -- weight decay penalty norm (1 or 2)
%                   decayRate         -- penalty on the weights
            %
            %   Similary, training Opts support the following fields:
            %       lRate       -- learning rate
            %       nIter       -- number of gradient iterations
            %       momentum    -- gradient moementum
            %       batchSz = 100;         % # of training points per batch
            %               lRate     -- coefficient on the gradient update
            %               decayNorm -- weight decay penalty norm (1 or 2)
            %               decayRate -- penalty on the weights
            %
            %               batchSz   -- size of the batches [optional]
            %               lRate     -- coefficient on the gradient update
            %               decayNorm -- weight decay penalty norm (1 or 2)
            %               decayRate -- penalty on the weights
            
            obj.nVis         = nVis;
            obj.nHid         = nHid;
            
            if ~isfield(pretrainOpts, 'nGS')
                pretrainOpts.nGS = 1;
            end     
            if ~isfield(pretrainOpts, 'dropVis')
                pretrainOpts.dropVis = 0;
            end
            if ~isfield(pretrainOpts, 'dropHid')
                pretrainOpts.dropHid = 0;
            end
            if ~isfield(pretrainOpts, 'sampleVis')
                pretrainOpts.sampleVis = false;
            end
            if ~isfield(pretrainOpts, 'sampleHid')
                pretrainOpts.sampleHid = true;
            end
            if ~isfield(pretrainOpts, 'momentum')
                pretrainOpts.momentum = 0;
            end
            if ~isfield(pretrainOpts, 'momentumChangeDif')
                pretrainOpts.momentumChange = 0;
            end
            
            if ~isfield(pretrainOpts, 'decayNorm')
                pretrainOpts.decayNorm = -1;
            end
            obj.pretrainOpts = pretrainOpts;
            
            if ~isfield(trainOpts, 'decayNorm')
                trainOpts.decayNorm = -1;
            end
            obj.trainOpts    = trainOpts;
            
            % Initializing weights 'à la Bengio'
            if ~isfield(trainOpts,'idStartWeights')
                range = sqrt(6/(obj.nVis + obj.nHid));
                obj.W = 2 * range * (rand(nVis, nHid) - .5);
            else
                obj.W = eye(nVis, nHid);
            end
            obj.b = zeros(nVis, 1);
            obj.c = zeros(nHid, 1);
            if isfield(trainOpts,'negStartBias')
                obj.b = obj.b -1;
                obj.c = obj.c -1;
            end
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
        function S = insize(self)
            S = self.nVis;
        end
        
        function S = outsize(self)
            S = self.nHid;
        end

        function [Y, A] = compute(self, X)
            if nargout == 2 && isfield(self.trainOpts, 'dropout')
                A.mask = rand(self.nVis, 1) > self.trainOpts.dropout;
                X = bsxfun(@times, X, A.mask ./ (1 - self.trainOpts.dropout));
            end
            Y = self.vis2hidprob(X);
            if nargout > 1
                A.x = X;
                A.s = Y;
            end
        end

        function [] = pretrain(self, X,preve)
            nObs = size(X, 2);
            opts = self.pretrainOpts;
            dWold   = zeros(size(self.W));
            dbold   = zeros(size(self.b));
            dcold   = zeros(size(self.c));
            act     = zeros(self.nHid, 1); % mean activity of hidden units
            msreold = 0;
            for epoch = 1:opts.nEpochs
                
                shuffle  = randperm(nObs);
                if isfield(self.trainOpts, 'cutTraining')
                    e = preve+epoch;
                else
                    e = epoch;
                end
                % Batch loop
                for batchBeg = 1:opts.batchSz:nObs
                    bind  = shuffle(batchBeg : min(nObs, ...
                        batchBeg + opts.batchSz -1));
                                        
                    % Gibbs sampling
                    [dW, db, dc, hid] = self.cd(X(:,bind));
                    
                    % Activity estimation (Hinton 2010)
                    if isfield(opts, 'sparsity') && e > opts.wDecayDelay
                        act = .9 * act + .1 * mean(hid, 2);
                    end
                    
                    % Hidden layer selectivity
                    if isfield(opts, 'selectivity') && e > opts.wDecayDelay
                        err = mean(hid, 1) - opts.selectivity;
                        ds  = bsxfun(@times, hid .* (1 - hid), err);
                        dW  = dW + opts.selectivityGain * X(:,bind) * ds' / nObs;
                        dc  = dc + mean(ds, 2);
                    end
                    
                    % Weight decay
                    if isfield(opts, 'wPenalty') && e > opts.wDecayDelay
                        dW = dW + opts.wPenalty * self.W;
                        db = db + opts.wPenalty * self.b;
                        dc = dc + opts.wPenalty * self.c;
                    end
                    
                    % Momentum
                    dW = dWold * opts.momentum + (1 - opts.momentum) * dW;
                    db = dbold * opts.momentum + (1 - opts.momentum) * db;
                    dc = dcold * opts.momentum + (1 - opts.momentum) * dc;
                    
                    % Apply gradient
                    self.W = self.W - opts.lRate * dW;
                    self.b = self.b - opts.lRate * db;
                    self.c = self.c - opts.lRate * dc;
                    if opts.decayNorm == 2
                         self.W = self.W - opts.lRate * opts.decayRate * self.W;
                   elseif opts.decayNorm == 1
                       self.W = self.W - opts.lRate * opts.decayRate * sign(self.W);
                    end
                    % Save gradient
                    dWold = dW;
                    dbold = db;
                    dcold = dc;
                end
                
                % Unit-wise sparsity (Hinton 2010)
                if isfield(opts, 'sparsity') && e > opts.wDecayDelay
                    dc = opts.lRate * opts.sparseGain * (act - opts.sparsity);
                    self.W = bsxfun(@minus, self.W, dc');
                    self.c = self.c - dc;
                end
                
                % Report
                if isfield(opts, 'displayEvery') ...
                        && mod(e, opts.displayEvery) == 0
                    % Reconstruct input samples
                    R = self.hid2vis(self.vis2hidprob(X));
                    % Mean square reconstruction error
                    msre = sqrt(mean(mean((R - X) .^2)));
                    fprintf('%03d , msre = %f\n', e, msre);
                    if isfield (opts,'momentumChangeDif') && ...
                        abs(msreold-msre) < opts.momentumChangeDif ...
                        && opts.momentum < opts.momentumChange
                        disp('Momentum change')
                        opts.momentum = opts.momentumChange;
                    end
                    msreold = msre;
                    
                   
                end
                if isfield(opts, 'displayWeights') ...
                        && mod(e, opts.displayWeights) == 0
                    weights = self.W;
                    figure(4);
                    [~, order] = sort(sum(weights .^2), 'descend');
                    colormap gray
                    for i = 1:20
                        subplot(5, 4, i);
                        imagesc(reshape(weights(:, order(i)), 6, 6));
                        axis image
                        axis off
                        
                    end
                    drawnow
                end
                if isfield(opts, 'calcHist') ...
                        && mod(e, opts.calcHist) == 0
                    % Create weight histogram
                    h1 = figure(1);
                    set(h1,'Visible','on');
                    disp('Creating weight histogram');

                    histogram(reshape(self.W,[numel(self.W),1]))
                    h2 = figure(2);
                    set(h2,'Visible','on');
                    histogram(reshape(dWold,[numel(dWold),1]))
                    drawnow
                end
                if isfield(opts, 'calcAct') ...
                        && mod(e, opts.calcAct) == 0
                    % Create weight histogram
                    disp('Creating activity histogram');
                    h3 = figure(3);
                    set(h3,'Visible','on');
                    histogram(act)
                    drawnow;
                end
                opts.lRate = opts.lRate*opts.lRateDecay;
            end
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            % backprop implementation of AbstractNet.backprop
            %   inErr = backprop(self, A, outErr, opts)
            %
            %   A      -- forward pass data as return by compute
            %   outErr -- network output error derivative w.r.t. output
            %             neurons stimulation
            %
            %   inErr  -- network output error derivative w.r.t. neurons
            %             outputs (not activation)
            
            % Gradient computation
            ds     = A.s .* (1- A.s);
            delta  = (outErr .* ds);
            G.dW   = A.x * delta';
            G.dc   = sum(delta, 2);
            
            % Error backpropagation
            inErr = self.W * delta;
            
            % Dropout
            if isfield(self.trainOpts, 'dropout')
                G.dW  = bsxfun(@times, G.dW, A.mask);
                inErr = bsxfun(@times, inErr, A.mask);
            end
        end
        
        function [] = gradientupdate(self, G)
            opts = self.trainOpts;
            % Gradient update
            self.W = self.W - opts.lRate * G.dW;
            self.c = self.c - opts.lRate * G.dc;
            
            % Weight decay
            if opts.decayNorm == 2
                self.W = self.W - opts.lRate * opts.decayRate * self.W;
            elseif opts.decayNorm == 1
                self.W = self.W - opts.lRate * opts.decayRate * sign(self.W);
            end
        end
        
        % Methods ----------------------------------------------------------- %
        
        function H = vis2hidbin(self, X)
            % Change to original code : add > rand() as suggested in
            % Hinton's guide to training such a system
            H = RBM.sigmoid(bsxfun(@plus, (X' * self.W)', self.c)) > rand();
        end
        
        function H = vis2hidprob(self, X)
            H = RBM.sigmoid(bsxfun(@plus, (X' * self.W)', self.c));
        end
        
        function V = hid2vis(self, H)
            V = RBM.sigmoid(bsxfun(@plus, self.W * H, self.b));
        end
        
        function [dW, db, dc, hid0] = cd(self, X)
            % CD Contrastive divergence (Hinton's CD(k))
            %   [dW, db, dc, act] = cd(self, X) returns the gradients of
            %   the weihgts, visible and hidden biases using Hinton's
            %   approximated CD. The sum of the average hidden units
            %   activity is returned in act as well.
            opts = self.pretrainOpts;
            
            nObs = size(X, 2);
            vis0 = X;
            hid0 = self.vis2hidbin(vis0);
            
            % Dropout masks
            if opts.dropHid > 0
                hmask = rand(size(hid0)) < opts.dropHid;
            end
            if opts.dropVis > 0
                vmask = rand(size(X)) < opts.dropVis;
            end

            hid = hid0;
            for k = 1:opts.nGS
                if opts.sampleHid % sampling ?
                    hid = hid > rand(size(hid));
                end
                if opts.dropHid > 0 % Dropout?
                    hid = hid .* hmask / (1 - opts.dropHid);
                end
                vis = self.hid2vis(hid);
                if opts.sampleVis % sampling ?
                    vis = vis > rand(size(vis));
                end
                if opts.dropVis > 0 % Dropout?
                    vis = vis .* vmask / (1 - opts.dropVis);
                    % TODO keep non masked visibles for CD but mask for hid
                    % computation.
                end
                if k == opts.nGS
                    hid = self.vis2hidprob(vis);
                else
                    hid = self.vis2hidbin(vis);
                end
                
            end
            
            dW      = - (vis0 * hid0' - vis * hid') / nObs;
            dc      = - (sum(hid0, 2) - sum(hid, 2)) / nObs;
            db      = - (sum(vis0, 2) - sum(vis, 2)) / nObs;
        end % cd(self, X)
        
        function  Images = generateFromRandom(self, nbout, nbiter)
            Images = zeros(self.nVis,nbout);
            for i =1:nbout
                vis0 = rand(self.nVis,1);
                hid0 = self.vis2hidprob(vis0);

                hid = hid0;
                vis = vis0;
                for k = 1:nbiter
                    vis = self.hid2vis(hid);
                    hid = self.vis2hidprob(vis);
                end
                Images(:,i) = vis;
            end

        end

        function msre = recErr(self,X)
              R = self.hid2vis(self.vis2hidprob(X));
              msre = sqrt(mean(mean((R - X) .^2)));

        end
    end % methods
    
    methods(Static)
        
        function p = sigmoid(X)
            p = 1./(1 + exp(-X));
        end
        
    end % methods(Static)
    
end % classdef RBM
