function [Kij] = cmpKij(LX)
    dist = distance(LX);
    %Kij = (1/sqrt(delta))*exp(-dist/delta);
    Kij = exp(-dist);
end