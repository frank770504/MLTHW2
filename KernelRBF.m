function [ out ] = KernelRBF( x1, x2 , gamma)
%KERNELRBF Summary of this function goes here
%   Detailed explanation goes here
    out = exp( -gamma*norm(x1 - x2,2));

end

