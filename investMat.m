%for k=20
fprintf('Flag  |  relres  |  iter\n');
fprintf('--------------------------\n');
for k=1:3
%for k=1
    % Load objects
    sk = sprintf('M_%d.mat', k);
    load(sk);

    % Determine some quantities
    d = size(B,1);
    ndofs = length(idx_inner);
    nW = size(M,1);
    %fprintf('Ndofs: %d\n', ndofs)
    %fprintf('Fraction of nonzeros: %f\n', nnz(Prec)/ prod(size(Prec)))

    % Check for zero coefficients
    il = [];
    jl = []; 
    for i = 1:d
        for j = 1:d
            if C{i,j} ~= -1e15
                il = [il,i];
                jl = [jl,j];
            end
        end
    end
    nnzs = length(il);


    % Set up block system for direct solve
    blk11 = blkdiag(M, M, M);
    blk13 = [];
    blk21 = [];
    for l = 1:nnzs
        blk13 = [blk13; -C{il(l),jl(l)}];
        blk21 = [blk21, -B{il(l),jl(l)}];
    end
    %blk13 = -[C{1,1}; C{2,2}; C{3,3}];
    %blk21 = -[B{1,1}, B{2,2}, B{3,3}];
    blk22 = M;
    blk32 = 0;
    for l=1:d
        blk32 = blk32 + C{l,l}';
    end
    Clapl = blk32';
    %blk32 = (C{1,1} + C{2,2} + C{3,3})';
    blk33 = S;
    
    % Determine correct sizes of zero matrizes
    blk12 = sparse(nW*nnzs,nW);
    blk23 = sparse(nW,nW);
    blk31 = sparse(nW,nW*nnzs);

    Blk = [blk11, blk12, blk13;
        blk21, blk22, blk23;
        blk31, blk32, blk33];

    i2a = @(S) S(:,idx_inner +1);
    i2i = @(S) S(idx_inner + 1,idx_inner +1);
    b2a = @(S) S(:, idx_bnd + 1);
    b2i = @(S) S(idx_bnd + 1, idx_bnd + 1);

    [L, U] = lu(M);

    % Deterime rhs
    g_bc = g(idx_bnd + 1)';
    Dv = 0;
    for l = 1:nnzs
        Dv = Dv + B{il(l), jl(l)} * ( U \ (L \ (b2a(C{il(l), jl(l)}) * g_bc)));
    end
    M_gc = U \ (L \ Dv);
    rhs = i2a(Clapl)' * ( U \ (L \f') - M_gc);
    
    Dv = @(x) 0;
    for l = 1:nnzs
        Dv = @(x) Dv(x) + B{il(l),jl(l)} * (U \ ( L \ ( i2a(C{il(l),jl(l)}) * x)));
    end
    x0 = zeros(ndofs,1);
    Sx = @(x) i2a(Clapl)' * ( U \ ( L \ (Dv(x)) ) )+ i2i(S) * x;
    
    % Test various preconditioners
    % Use inverse of lumped mass matrix
    %variant = 'Mlumpinv';
    %variant = 'Mdiaginv';
    %variant = 'plainMlump';
    %variant = 'plainMdiag';
    %variant = 'noPrec';
    variant = 'ilu';
    tic
    switch variant
    case 'Mlumpinv'
        Mlumpinv = spdiags(1./sum(M,2),[0],nW,nW);
        P = 0;
        for l = 1:nnzs
            P = P + B{il(l),jl(l)} * Mlumpinv * i2a(C{il(l),jl(l)});
        end
        P = i2a(Clapl)' * Mlumpinv * P + i2i(S);
        [PL,PU] = lu(P);
    case 'Mdiaginv'
        Mdiaginv = spdiags(1./diag(M),[0],nW,nW);
        P = 0;
        for l = 1:nnzs
            P = P + B{il(l),jl(l)} * Mdiaginv * i2a(C{il(l),jl(l)});
        end
        P = i2a(Clapl)' * Mdiaginv * P + i2i(S);
        [PL,PU] = lu(P);
    case 'plainMlump'
        Mlumpinv = spdiags(1./sum(M,2),[0],nW,nW);
        P = i2i(Mlumpinv);
        [PL,PU] = lu(P);
    case 'plainMdiag'
        Mdiaginv = spdiags(diag(M),[0],nW,nW);
        P = i2i(Mdiaginv);
        [PL,PU] = lu(P);
    case 'noPrec'
        P = speye(ndofs);
        [PL,PU] = lu(P);
    case 'ilu'
        P = 0;
        Mdiaginv = spdiags(1./diag(M),[0],nW,nW);

        for l = 1:nnzs
            Bij = B{il(l), jl(l)};
            Bijdiag = spdiags(diag(Bij),[0],nW,nW);
            P = P + Bijdiag * Mdiaginv * i2a(C{il(l),jl(l)});
        end
        P = i2a(Clapl)' * Mdiaginv * P + i2i(S);
        [PL,PU] = lu(P);
        %setup.type = 'ilutp';
        %P = sparse(P);
        %[PL,PU] = ilu(P, setup);
    case 'ilu2'
        P = 0;
        Mdiaginv = spdiags(1./diag(M),[0],nW,nW);
        Bsum = 0;
        for l = 1:nnzs
            Bij = B{il(l), jl(l)};
            Bijdiag = spdiags(diag(Bij),[0],nW,nW);
            Bsum = Bsum + Bijdiag;
            %P = P + Bijdiag * Mdiaginv * i2a(C{il(l),jl(l)});
        end

        P = i2a(Clapl)' * Mdiaginv * Bsum * Mdiaginv * i2a(Clapl) + i2i(S);
        [PL,PU] = lu(P);
        %setup.type = 'ilutp';
        %P = sparse(P);
        %[PL,PU] = ilu(P, setup);
    end
    
    [x, flag, relres, iter] = gmres(Sx,rhs,[],1e-10,min(1000,ndofs), PL, PU, x0);
    fprintf('   %d  |  %6.4f  |  %4d  (%s)\n', flag, relres, iter(2), variant);
    toc
end
