function [distance] = cost_J(J0, J_rate, A, V, points)

J0_all=J0 .* J_rate;
cross0_all=cross((J0_all - A), V);
distance_all = sqrt(cross0_all(:,1,:).^2 + cross0_all(:,2,:).^2 + cross0_all(:,3,:).^2);

d_A0_I=sum((repmat(A,[size(points,1),1,size(points,3)])-points).^2,2);
d_A0_J=sum((repmat(A,[size(points,1),1,size(points,3)])-J0_all).^2,2);
t_result=d_A0_I./d_A0_J;
[t_result_sort, sortedIndices] = sort(t_result,1, 'descend');

for i = 1:size(distance_all, 3)
    distance(:,:,i)=distance_all(sortedIndices(1:2,:,i), :, i);
end
distance=repmat(distance,[2,1,1]);

end

