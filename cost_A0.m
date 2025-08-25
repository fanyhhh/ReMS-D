function [distance] = cost(xyz, A, B, C, D)

distance = abs(A * xyz(1) + B * xyz(2) + C * xyz(3) + D);

end

