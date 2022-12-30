#Modificed From athena++
#ONLY WORK FOR uniform CERT COOR AND HD VERSION

#Module : 2nd order reconstruction function
#Date   : 17 Dec 2022
#Author : Ka Wai HO

# dispath function
PiecewiseParabolicX₁!(w, wl, wr, grid::AbstractGrid) = PiecewiseParabolicXᵢ!(w, wl, wr, grid; dir="1")
PiecewiseParabolicX₂!(w, wl, wr, grid::AbstractGrid) = PiecewiseParabolicXᵢ!(w, wl, wr, grid; dir="2")
PiecewiseParabolicX₃!(w, wl, wr, grid::AbstractGrid) = PiecewiseParabolicXᵢ!(w, wl, wr, grid; dir="3")

function PiecewiseParabolicXᵢ!(w, wl, wr,
                            grid::AbstractGrid; dir="x")
  T = eltype(grid)
  # Get back the infromation of all dimonions
  is,ie = grid.x1.is::Int, grid.x1.ie::Int;
  js,je = grid.x2.js::Int, grid.x2.je::Int;
  ks,ke = grid.x3.ks::Int, grid.x3.ke::Int;
  Nhydro = grid.Nhydro::Int
  
  # CUDA threads & PLM function setup
  if dir=="1"
    threads = ( 32, 16, 1)
    PPMXᵢ_CUDA! = PPMX₁_CUDA!
  elseif dir == "2"
    threads = ( 32, 16, 1)
    PPMXᵢ_CUDA! = PPMX₂_CUDA!
  elseif dir == "3"
    threads = ( 32, 16, 1)
    PPMXᵢ_CUDA! = PPMX₃_CUDA!
  end
  blocks   = (ceil(Int,size(w,1)/threads[1]), ceil(Int,size(w,2)/threads[2]), ceil(Int,size(w,3)/threads[3])) 
  for n = 1:Nhydro
     w_3D = (@view  w[:,:,:,n])::CuArray{T,3} 
    wl_3D = (@view wl[:,:,:,n])::CuArray{T,3}
    wr_3D = (@view wr[:,:,:,n])::CuArray{T,3}
    @cuda blocks = blocks threads = threads PPMXᵢ_CUDA!(w_3D, wl_3D, wr_3D,
                                                        is, ie, js, je, ks, ke)
  end
  return nothing
end

function PPMX₁_CUDA!(w, wl, wr,
                     is, ie, js, je, ks, ke)
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 

  # we will need c1i_i c1i_im1 c1i_ip1 in the future
  T  = eltype(w)
  local c1i = c2i = c3i = c4i = T(0.5);
  local c5i =  T(1.0/6.0);
  local c6i = -T(1.0/6.0);
  local  C2 =  T(1.25);


  if k ∈ (ks:ke) && j ∈ (js:je) && i ∈ (is-2:ie+2)
    @inbounds q_i   = w[i  ,j,k]
    @inbounds q_im2 = w[i-2,j,k]
    @inbounds q_im1 = w[i-1,j,k]
    @inbounds q_ip1 = w[i+1,j,k]
    @inbounds q_ip2 = w[i+2,j,k]

    qa =   q_i - q_im1;
    qb = q_ip1 -   q_i;
    dd_im1 = c1i*qa + c2i*(q_im1 - q_im2);
    dd     = c1i*qb + c2i*qa;
    dd_ip1 = c1i*(q_ip2 - q_ip1) + c2i*qb;

    # Approximate interface average at i-1/2 and i+1/2 using PPM (CW eq 1.6)
    # KGF: group the biased stencil quantities to preserve FP symmetry
    dph     = (c3i*q_im1 + c4i*q_i) +  (c5i*dd_im1 + c6i*dd);
    dph_ip1 = (c3i*q_i + c4i*q_ip1) + (c5i*dd + c6i*dd_ip1 );

    d2qc_im1 = q_im2 + q_i   - 2.0*q_im1;#
    d2qc     = q_im1 + q_ip1 - 2.0*q_i  ;# // (CD eq 85a) (no 1/2)
    d2qc_ip1 = q_i   + q_ip2 - 2.0*q_ip1;#
          
    # i - 1/2
    qa_tmp = dph - q_im1; #// (CD eq 84a)
    qb_tmp = q_i - dph;     #// (CD eq 84b)
    #// KGF: add the off-centered quantities first to preserve FP symmetry
    qa = 3.0*(q_im1 + q_i  - 2.0*dph);  /# (CD eq 85b)
    qb = d2qc_im1;    # (CD eq 85a) (no 1/2)
    qc = d2qc;   # (CD eq 85c) (no 1/2)
    qd = 0.0;

    if (sign(qa) == sign(qb) && sign(qa) == sign(qc))
      qd = sign(qa)* min(C2*abs(qb), min(C2*abs(qc), abs(qa)));
    end

    dph_tmp = 0.5*(q_im1 + q_i) - qd/6.0;
    #// Local extrema detected at i-1/2 face
    dph = qa_tmp*qb_tmp < 0.0 ? dph_tmp : dph
      
    # i+1/2
    qa_tmp = dph_ip1 - q_i;         #// (CD eq 84a)
    qb_tmp = q_ip1   - dph_ip1;   #// (CD eq 84b)
    #// KGF: add the off-centered quantities first to preserve FP symmetry
    qa = 3.0*(q_i+ q_ip1 - 2.0*dph_ip1);  #// (CD eq 85b)
    qb = d2qc;            #// (CD eq 85a) (no 1/2)
    qc = d2qc_ip1;        #// (CD eq 85c) (no 1/2)
    qd = 0.0;
    if (sign(qa) == sign(qb) && sign(qa) == sign(qc))
      qd = sign(qa)* min(C2*abs(qb), min(C2*abs(qc), abs(qa)));
    end
    dphip1_tmp = 0.5*(q_i + q_ip1) - qd/6.0;
    #// Local extrema detected at i+1/2 face
    dph_ip1 = qa_tmp*qb_tmp < 0.0 ? dphip1_tmp : dph_ip1;  

    d2qf = 6.0*(dph + dph_ip1 - 2.0*q_i);

    qminus = dph
    qplus  = dph_ip1

    dqf_minus = q_i - qminus #// (CS eq 25)
    dqf_plus = qplus - q_i
            
    #//--- Step 4a. -----------------------------------------------------------------------
    #// For uniform Cartesian-like coordinate: apply CS limiters to parabolic interpolant
    qa_tmp = dqf_minus*dqf_plus
    qb_tmp = (q_ip1 - q_i)*(q_i- q_im1);

    qa = d2qc_im1
    qb = d2qc
    qc = d2qc_ip1
    qd = d2qf
    qe = 0.0;
    if (sign(qa) == sign(qb) && sign(qa) == sign(qc) && sign(qa) == sign(qd))
    #// Extrema is smooth
      qe = sign(qd)* min(min(C2*abs(qa), C2*abs(qb)), min(C2*abs(qc),abs(qd))); # (CS eq 22)
    end

    #// Check if 2nd derivative is close to roundoff error
    qa = max(abs(q_im1), abs(q_im2))
    qb = max(max(abs(q_i), abs(q_ip1), abs(q_ip2)));

    #// Limiter is not sensitive to roundoff. Use limited ratio (MC eq 27)
    rho = abs(qd) > (1.0e-12)*max(qa,qb) ? 0.0 : qe/qd

    tmp_m  = q_i - rho*dqf_minus;
    tmp_p  = q_i + rho*dqf_plus;
    tmp2_m = q_i - 2.0*dqf_plus;
    tmp2_p = q_i + 2.0*dqf_minus;

    #// Check for local extrema
    if ((qa_tmp <= 0.0 || qb_tmp <= 0.0))
      #// Check if relative change in limited 2nd deriv is > roundoff
      if (rho <= (1.0 - (1.0e-12)))
        #// Limit smooth extrema
        qminus = tmp_m; #// (CS eq 23)
         qplus = tmp_p;
      end
      #// No extrema detected
    else
      #// Overshoot i-1/2,R / i,(-) state
      if (abs(dqf_minus) >= 2.0*abs(dqf_plus))
        qminus = tmp2_m;
      end
      #// Overshoot i+1/2,L / i,(+) state
      if (abs(dqf_plus) >= 2.0*abs(dqf_minus))
        qplus = tmp2_p;
      end
    end               
    @inbounds wl[i+1,j,k] =  qplus
    @inbounds wr[i  ,j,k] = qminus
   end
  return nothing
end

function PPMX₂_CUDA!( w, wl, wr,
                     is, ie, js, je, ks, ke)
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 

  # we will need c1i_i c1i_im1 c1i_ip1 in the future
  T = eltype(w)
  local c1i = c2i = c3i = c4i = T(0.5)
  local c5i =  T(1.0/6.0)
  local c6i = -T(1.0/6.0)
  local  C2 =  T(1.25)

  if k ∈ (ks:ke) && j ∈ (js-2:je+2)  && i ∈ (is:ie)
      @inbounds q_i   = w[i,j  ,k]
      @inbounds q_im2 = w[i,j-2,k]
      @inbounds q_im1 = w[i,j-1,k]
      @inbounds q_ip1 = w[i,j+1,k]
      @inbounds q_ip2 = w[i,j+2,k]

      # step 1
      qa =   q_i - q_im1;
      qb = q_ip1 -   q_i;
      dd_im1 = c1i*qa + c2i*(q_im1 - q_im2);
      dd     = c1i*qb + c2i*qa;
      dd_ip1 = c1i*(q_ip2 - q_ip1) + c2i*qb;

      # Approximate interface average at i-1/2 and i+1/2 using PPM (CW eq 1.6)
      # KGF: group the biased stencil quantities to preserve FP symmetry
      dph = (c3i*q_im1 + c4i*q_i) +  (c5i*dd_im1 + c6i*dd);
      dph_ip1 = (c3i*q_i + c4i*q_ip1) + (c5i*dd + c6i*dd_ip1 );

      d2qc_im1 = q_im2 + q_i   - 2.0*q_im1;#
      d2qc     = q_im1 + q_ip1 - 2.0*q_i  ;# // (CD eq 85a) (no 1/2)
      d2qc_ip1 = q_i   + q_ip2 - 2.0*q_ip1;#
            
    
      # i - 1/2
      qa_tmp = dph - q_im1;  #// (CD eq 84a)
      qb_tmp = q_i - dph;    #// (CD eq 84b)
      #// KGF: add the off-centered quantities first to preserve FP symmetry
      qa = 3.0*(q_im1 + q_i  - 2.0*dph);  /# (CD eq 85b)
      qb = d2qc_im1;    # (CD eq 85a) (no 1/2)
      qc = d2qc;   # (CD eq 85c) (no 1/2)
      qd = 0.0;

      if (sign(qa) == sign(qb) && sign(qa) == sign(qc))
        qd = sign(qa)* min(C2*abs(qb), min(C2*abs(qc), abs(qa)));
      end

      dph_tmp = 0.5*(q_im1 + q_i) - qd/6.0;
      #// Local extrema detected at i-1/2 face
      dph = qa_tmp*qb_tmp < 0.0 ? dph_tmp : dph
      
      # i+1/2
      qa_tmp = dph_ip1 - q_i;         #// (CD eq 84a)
      qb_tmp = q_ip1   - dph_ip1;   #// (CD eq 84b)
      #// KGF: add the off-centered quantities first to preserve FP symmetry
      qa = 3.0*(q_i+ q_ip1 - 2.0*dph_ip1);  #// (CD eq 85b)
      qb = d2qc;            #// (CD eq 85a) (no 1/2)
      qc = d2qc_ip1;        #// (CD eq 85c) (no 1/2)
      qd = 0.0;
      if (sign(qa) == sign(qb) && sign(qa) == sign(qc))
        qd = sign(qa)* min(C2*abs(qb), min(C2*abs(qc), abs(qa)));
      end
      dphip1_tmp = 0.5*(q_i + q_ip1) - qd/6.0;
      #// Local extrema detected at i+1/2 face
      dph_ip1 = qa_tmp*qb_tmp < 0.0 ? dphip1_tmp : dph_ip1;  

      d2qf = 6.0*(dph + dph_ip1 - 2.0*q_i);

      qminus = dph
      qplus  = dph_ip1

      dqf_minus = q_i - qminus #// (CS eq 25)
       dqf_plus = qplus - q_i
            
    # //--- Step 4a. -----------------------------------------------------------------------
    #// For uniform Cartesian-like coordinate: apply CS limiters to parabolic interpolant
      qa_tmp = dqf_minus*dqf_plus
      qb_tmp = (q_ip1 - q_i)*(q_i- q_im1);

      qa = d2qc_im1
      qb = d2qc
      qc = d2qc_ip1
      qd = d2qf
      qe = 0.0;
      if (sign(qa) == sign(qb) && sign(qa) == sign(qc) && sign(qa) == sign(qd))
      #// Extrema is smooth
        qe = sign(qd)* min(min(C2*abs(qa), C2*abs(qb)), min(C2*abs(qc),abs(qd))); # (CS eq 22)
      end

      #// Check if 2nd derivative is close to roundoff error
      qa = max(abs(q_im1), abs(q_im2))
      qb = max(max(abs(q_i), abs(q_ip1), abs(q_ip2)));

      #// Limiter is not sensitive to roundoff. Use limited ratio (MC eq 27)
      rho = abs(qd) > (1.0e-12)*max(qa,qb) ? 0.0 : qe/qd

      tmp_m  = q_i - rho*dqf_minus;
      tmp_p  = q_i + rho*dqf_plus;
      tmp2_m = q_i - 2.0*dqf_plus;
      tmp2_p = q_i + 2.0*dqf_minus;

      #// Check for local extrema
      if ((qa_tmp <= 0.0 || qb_tmp <= 0.0))
          #// Check if relative change in limited 2nd deriv is > roundoff
          if (rho <= (1.0 - (1.0e-12)))
            #// Limit smooth extrema
            qminus = tmp_m; #// (CS eq 23)
             qplus = tmp_p;
          end
          #// No extrema detected
      else
          #// Overshoot i-1/2,R / i,(-) state
          if (abs(dqf_minus) >= 2.0*abs(dqf_plus))
            qminus = tmp2_m;
          end
          #// Overshoot i+1/2,L / i,(+) state
          if (abs(dqf_plus) >= 2.0*abs(dqf_minus))
            qplus = tmp2_p;
          end
      end
                
      @inbounds wl[i,j+1,k] =  qplus
      @inbounds wr[i,j  ,k] = qminus
  end
    return nothing
end

function PPMX₃_CUDA!(w, wl, wr,
                     is, ie, js, je, ks, ke)
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 

  # we will need c1i_i c1i_im1 c1i_ip1 in the future
  T = eltype(w)
  local c1i = c2i = c3i = c4i = T(0.5)
  local c5i =  T(1.0/6.0)
  local c6i = -T(1.0/6.0)
  local  C2 =  T(1.25)

  if k ∈ (ks-2:ke+2) && j ∈ (js:je)  && i ∈ (is:ie)
    @inbounds q_i   = w[i,j,k  ]
    @inbounds q_im2 = w[i,j,k-2]
    @inbounds q_im1 = w[i,j,k-1]
    @inbounds q_ip1 = w[i,j,k+1]
    @inbounds q_ip2 = w[i,j,k+2]

    qa =   q_i - q_im1;
    qb = q_ip1 -   q_i;
    dd_im1 = c1i*qa + c2i*(q_im1 - q_im2);
    dd     = c1i*qb + c2i*qa;
    dd_ip1 = c1i*(q_ip2 - q_ip1) + c2i*qb;

    # Approximate interface average at i-1/2 and i+1/2 using PPM (CW eq 1.6)
    # KGF: group the biased stencil quantities to preserve FP symmetry
    dph = (c3i*q_im1 + c4i*q_i) +  (c5i*dd_im1 + c6i*dd);
    dph_ip1 = (c3i*q_i + c4i*q_ip1) + (c5i*dd + c6i*dd_ip1 );

    d2qc_im1 = q_im2 + q_i   - 2.0*q_im1;#
    d2qc     = q_im1 + q_ip1 - 2.0*q_i  ;# // (CD eq 85a) (no 1/2)
    d2qc_ip1 = q_i   + q_ip2 - 2.0*q_ip1;#
          
    # i - 1/2
    qa_tmp = dph - q_im1; #// (CD eq 84a)
    qb_tmp = q_i - dph;     #// (CD eq 84b)
    #// KGF: add the off-centered quantities first to preserve FP symmetry
    qa = 3.0*(q_im1 + q_i  - 2.0*dph);  /# (CD eq 85b)
    qb = d2qc_im1;    # (CD eq 85a) (no 1/2)
    qc = d2qc;   # (CD eq 85c) (no 1/2)
    qd = 0.0;

    if (sign(qa) == sign(qb) && sign(qa) == sign(qc))
      qd = sign(qa)* min(C2*abs(qb), min(C2*abs(qc), abs(qa)));
    end

    dph_tmp = 0.5*(q_im1 + q_i) - qd/6.0;
    #// Local extrema detected at i-1/2 face
    dph = qa_tmp*qb_tmp < 0.0 ? dph_tmp : dph
    
    # i+1/2
    qa_tmp = dph_ip1 - q_i;         #// (CD eq 84a)
    qb_tmp = q_ip1   - dph_ip1;   #// (CD eq 84b)
    #// KGF: add the off-centered quantities first to preserve FP symmetry
    qa = 3.0*(q_i+ q_ip1 - 2.0*dph_ip1);  #// (CD eq 85b)
    qb = d2qc;            #// (CD eq 85a) (no 1/2)
    qc = d2qc_ip1;        #// (CD eq 85c) (no 1/2)
    qd = 0.0;
    if (sign(qa) == sign(qb) && sign(qa) == sign(qc))
      qd = sign(qa)* min(C2*abs(qb), min(C2*abs(qc), abs(qa)));
    end
    dphip1_tmp = 0.5*(q_i + q_ip1) - qd/6.0;
    #// Local extrema detected at i+1/2 face
    dph_ip1 = qa_tmp*qb_tmp < 0.0 ? dphip1_tmp : dph_ip1;  

    d2qf = 6.0*(dph + dph_ip1 - 2.0*q_i);

    qminus = dph
    qplus  = dph_ip1

    dqf_minus = q_i - qminus #// (CS eq 25)
     dqf_plus = qplus - q_i
          
  # //--- Step 4a. -----------------------------------------------------------------------
  #// For uniform Cartesian-like coordinate: apply CS limiters to parabolic interpolant
    qa_tmp = dqf_minus*dqf_plus
    qb_tmp = (q_ip1 - q_i)*(q_i- q_im1);

    qa = d2qc_im1
    qb = d2qc
    qc = d2qc_ip1
    qd = d2qf
    qe = 0.0;
    if (sign(qa) == sign(qb) && sign(qa) == sign(qc) && sign(qa) == sign(qd))
    #// Extrema is smooth
      qe = sign(qd)* min(min(C2*abs(qa), C2*abs(qb)), min(C2*abs(qc),abs(qd))); # (CS eq 22)
    end

    #// Check if 2nd derivative is close to roundoff error
    qa = max(abs(q_im1), abs(q_im2))
    qb = max(max(abs(q_i), abs(q_ip1), abs(q_ip2)));

    #// Limiter is not sensitive to roundoff. Use limited ratio (MC eq 27)
    rho = abs(qd) > (1.0e-12)*max(qa,qb) ? 0.0 : qe/qd

    tmp_m  = q_i - rho*dqf_minus;
    tmp_p  = q_i + rho*dqf_plus;
    tmp2_m = q_i - 2.0*dqf_plus;
    tmp2_p = q_i + 2.0*dqf_minus;

    #// Check for local extrema
    if ((qa_tmp <= 0.0 || qb_tmp <= 0.0))
        #// Check if relative change in limited 2nd deriv is > roundoff
        if (rho <= (1.0 - (1.0e-12)))
          #// Limit smooth extrema
          qminus = tmp_m; #// (CS eq 23)
           qplus = tmp_p;
        end
        #// No extrema detected
    else
        #// Overshoot i-1/2,R / i,(-) state
        if (abs(dqf_minus) >= 2.0*abs(dqf_plus))
          qminus = tmp2_m;
        end
        #// Overshoot i+1/2,L / i,(+) state
        if (abs(dqf_plus) >= 2.0*abs(dqf_minus))
          qplus = tmp2_p;
        end
    end
               
    @inbounds wl[i,j,k+1] =  qplus
    @inbounds wr[i,j,k  ] = qminus
  end
    return nothing
end