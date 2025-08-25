function [x, mask_finish] = newton_t(f, x0, tol, step, maxiter)
x = x0;

mask_finish=ones(size(x0));

for i = 1:maxiter
    fx = f(x);
    
    mask_finish(abs(fx) < tol)=0;
    if sum(mask_finish(:)) == 0
        break;
    end
    
    fpx = (f(x+step) - fx) / step;
    delta=real(fx ./ fpx);
    delta(isnan(delta)) = 0;
    delta(isinf(delta)) = 0;
    
    x = x - delta .* mask_finish;

    x=min(max(x,1e-3),1);

end
end