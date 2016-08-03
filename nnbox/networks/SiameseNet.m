classdef SiameseNet < AbstractNet & handle
    % SIAMESENET implements the siamese network pattern of network
    % replication and weight sharing as an AbstractNet
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        nNets;        % Number of replications
        net;          % actual network instance
        pretrainOpts; % supervized training options
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = SiameseNet(net, n, varargin)
            % obj = SIAMESENET(net, N) returns an instance of SIAMESENET
            % with N copies of the neural network net
            %
            % obj = SIAMESENET(net, N, 'skipPretrain') makes the network
            % ignore pretraining requests
            
            assert(isa(net, 'AbstractNet'), 'net must implement AbstractNet');
            assert(~iscell(net.outsize()) && ~iscell(net.insize()), ...
                'Grouped input or output not supported');
            if n == 1, warning('SiameseNet is useless with n = 1'); end
            
            obj.nNets = n;
            obj.net = net.copy();
            obj.pretrainOpts.skip = false;
            
            if ~isempty(varargin) && strcmp(varargin{1}, 'skipPretrain')
                obj.pretrainOpts.skip = true;
            end
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
        function s = insize(self)
            s    = cell(self.nNets, 1);
            s(:) = {self.net.insize()};
        end
        
        function s = outsize(self)
            s    = cell(self.nNets, 1);
            s(:) = {self.net.outsize()};
        end
        
        function [Y, A] = compute(self, X)
            if nargout == 1
                Y = [];
                for i = 1:self.nNets
                    Y = vertcat(Y,self.net.compute(cell2mat(X(i,:))));
                end
            else
                Y = [];
                A = [];
                for i = 1:self.nNets
                    [tmpY, tmpA] = self.net.compute(cell2mat(X(i,:)));
                    Y = vertcat(Y,tmpY);
                    A = [A tmpA];
                end
            end
        end
        
        function [] = pretrain(self, X)
            if ~self.pretrainOpts.skip
                X = cell2mat(reshape(X, 1, numel(X)));
                %TODO support multidimensional input
                self.net.pretrain(X);
            end
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            outErr = reshape(outErr,self.nNets,size(outErr,1)/self.nNets,size(outErr,2));
            G     = [];
            inErr = [];
            for c = 1:self.nNets
                [tmpG, tmpinErr] = self.net.backprop(A(:,c), reshape(outErr(c,:,:),size(outErr,2),size(outErr,3)));
                G = [G tmpG];
                inErr = vertcat(inErr,tmpinErr);
            end
        end
        
        function gradientupdate(self, G)
            for c = 1:self.nNets
                self.net.gradientupdate(G(:,c));
            end
        end
    end
    
    methods(Access = protected)
        
        % Copyable implementation ------------------------------------------- %
        
        % Override copyElement method
        function copy = copyElement(self)
            if self.pretrainOpts.skip
                copy = SiameseNet(self.net.copy(), self.nNets, 'skipPretrain');
            else
                copy = SiameseNet(self.net.copy(), self.nNets);
            end
        end
        
    end
    
end

