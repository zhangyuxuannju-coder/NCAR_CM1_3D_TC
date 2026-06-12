;this program is used to plot azimal averaged Vt .vs. radius
;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
;************************************************

begin

   ntimes       = 25
   sumlen       = 121
   ang          = 360
   nlevels      = 37
   f            = 5.0e-5
   g            = 9.81
   theta0       = 300.0
   dr           = 2000.0
   dz           = 500.0
   cp           = 1004.6

  r = new((/sumlen/),"float")
  z = new((/nlevels/),"float")

  do i = 1 , sumlen-1
     r(i) = 2000.0*i
  end do
  r(0) = 1000.0

  do k = 0 , nlevels-1
     z(k) = 500.0*k
  end do  

   theta = new((/nlevels,sumlen/),"float")    
   radi  = new((/nlevels,sumlen/),"float")
   rho   = new((/nlevels,sumlen/),"float")
   vt    = new((/nlevels,sumlen/),"float")

   do i = 0 , 36     
      theta  = fbinrecread("./SE-data/theta.ieee",0,(/nlevels,sumlen/),"float")
      radi   = fbinrecread("./SE-data/Q.ieee",0,(/nlevels,sumlen/),"float")  
      rho    = fbinrecread("./SE-data/rho.ieee",0,(/nlevels,sumlen/),"float") 
      vt     = fbinrecread("./SE-data/vgrad.ieee",0,(/nlevels,sumlen/),"float") 
   end do

   do i = 0 , 14
      radi(:,i) = ((1.0*i)/15.0)*radi(:,15)
   end do

   khi = new((/nlevels,sumlen/),"float")
   khi = 1.0/theta

   khi@_FillValue  = 9.96921e+36
   vt@_FillValue   = 9.96921e+36
   radi@_FillValue = 9.96921e+36
   rho@_FillValue  = 9.96921e+36

do i = 1 ,nlevels-1-1
   khi(i,:) = dim_avg_n(khi(i-1:i+1,:),0)
   vt(i,:)  = dim_avg_n(vt(i-1:i+1,:),0)  
   rho(i,:) = dim_avg_n(rho(i-1:i+1,:),0)
end do

do i = 1 ,sumlen-1-1
   khi(:,i) = dim_avg_n(khi(:,i-1:i+1),1)
   vt(:,i)  = dim_avg_n(vt(:,i-1:i+1),1) 
   rho(:,i) = dim_avg_n(rho(:,i-1:i+1),1)
end do

   ksi   = new((/nlevels,sumlen/),"float")
   ct    = new((/nlevels,sumlen/),"float")
   vor   = new((/nlevels,sumlen/),"float")
   iner  = new((/nlevels,sumlen/),"float")

do i = 0, sumlen-1
   ksi(:,i) = f + 2*vt(:,i)/r(i)
   ct(:,i)  = (vt(:,i)/r(i) + f) * vt(:,i)
end do

fc = 0.01

do i = 0 , sumlen-1

  if (i .eq. 0) then 
     vor(:,i) = f + vt(:,1)/r(1) + (vt(:,i+1)-vt(:,i))/(dr)
  else
     if (i .lt. sumlen-1) then
        vor(:,i) = f + vt(:,i)/r(i) + (vt(:,i+1)-vt(:,i-1))/(2*dr)
     else
        vor(:,i) = f + vt(:,i)/r(i) + (vt(:,i)-vt(:,i-1))/(dr)
     end if
  end if

end do

vor_min = min(vor)

;=========================================================
; 修正：vor < 0.01*f 设为 0.01*f
;=========================================================

do k = 0 , nlevels-1
  do i = 0 , sumlen-1

    if (vor(k,i) .lt. fc*f) then
        vor(k,i) = fc*f
    end if

  end do
end do

   iner = ksi*vor

      F = new((/nlevels,sumlen/),"float")
      Q = new((/nlevels,sumlen/),"float")
      Q = khi^2*radi

      F(:,0) = 1.0*g*(Q(:,1)-Q(:,0))/(dr)
   do i = 1 , sumlen-1-1
      F(:,i) = 1.0*g*(Q(:,i+1)-Q(:,i-1))/(2*dr)
   end do
      F(:,sumlen-1) = 1.0*g*(Q(:,sumlen-1)-Q(:,sumlen-1-1))/(dr)

  aa   = new((/nlevels,sumlen/),"float")
  bb   = new((/nlevels,sumlen/),"float")
  cc   = new((/nlevels,sumlen/),"float")
  cc1  = new((/nlevels,sumlen/),"float")
  cc2  = new((/nlevels,sumlen/),"float")
  ckhi = new((/nlevels,sumlen/),"float")
  ckhi = khi*ct

do k = 0 , nlevels-1
  if (k .eq. 0) then
    aa(k,:) = -1.0*g * (khi(k+1,:)-khi(k,:))/(dz)
    bb(k,:) = -2.0 * (ckhi(k+1,:) - ckhi(k,:))/(dz)
  else
    if (k .lt. nlevels-1) then
         aa(k,:) = -1.0*g * (khi(k+1,:)-khi(k-1,:))/(2*dz)
         bb(k,:) = -2.0 * (ckhi(k+1,:) - ckhi(k-1,:))/(2*dz)
    else 
         aa(k,:) = -1.0*g * (khi(k,:)-khi(k-1,:))/(dz)
         bb(k,:) = -2.0 * (ckhi(k,:) - ckhi(k-1,:))/(dz)
    end if
  end if
end do 

bb = bb*0.4
cc1 = khi*vor*ksi

do i = 0 , sumlen-1
   if (i .eq. 0) then
       cc2(:,i) = ct(:,i)*(khi(:,i+1)-khi(:,i))/dr
   else
     if (i .lt. sumlen-1) then
          cc2(:,i) = ct(:,i)*(khi(:,i+1)-khi(:,i-1))/(2*dr)
     else
            cc2(:,i) = ct(:,i)*(khi(:,i)-khi(:,i-1))/dr
     end if
   end if
end do 

cc = cc1+cc2

 ;-------  Solver Coefficient     -------
  A1   = new((/nlevels,sumlen/),"float")
  B1   = new((/nlevels,sumlen/),"float")
  C1   = new((/nlevels,sumlen/),"float")
  A    = new((/nlevels,sumlen/),"float")
  B    = new((/nlevels,sumlen/),"float")
  C    = new((/nlevels,sumlen/),"float")
  D    = new((/nlevels,sumlen/),"float")
  Da   = new((/nlevels,sumlen/),"float")
  Db   = new((/nlevels,sumlen/),"float")
  E    = new((/nlevels,sumlen/),"float")
  Ea   = new((/nlevels,sumlen/),"float")
  Eb   = new((/nlevels,sumlen/),"float")

do i = 0 , sumlen-1
   A1(:,i) = aa(:,i)/(r(i)*rho(:,i))
   B1(:,i) = bb(:,i)/(r(i)*rho(:,i))
   C1(:,i) = cc(:,i)/(r(i)*rho(:,i))
end do 


   A  = A1
   B  = B1
   C  = C1

;=========================================================
; elliptic 诊断（画图 + 输出）
;=========================================================

elliptic = A*C - (B*B)/4.0

print("==== ELLIPTIC CHECK ====")
print("min = " + min(elliptic))
print("max = " + max(elliptic))
print("count(<=0) = " + num(elliptic .le. 0))

do i = 0 , sumlen-1
   if (i .eq. 0) then
       Da(:,i) = (A1(:,i+1)-A1(:,i))/(dr)
       Ea(:,i) = (B1(:,i+1)-B1(:,i))/(2*dr)
   else
       if (i .lt. sumlen-1) then
           Da(:,i) = (A1(:,i+1)-A1(:,i-1))/(2*dr)
           Ea(:,i) = (B1(:,i+1)-B1(:,i-1))/(4*dr)
       else
           Da(:,i) = (A1(:,i)-A1(:,i-1))/(dr)
           Ea(:,i) = (B1(:,i)-B1(:,i-1))/(2*dr)
       end if
   end if 
end do 
         
do k = 0 , nlevels-1
   if (k .eq. 0) then
       Db(k,:) = (B1(k+1,:)-B1(k,:))/(2*dz)
       Eb(k,:) = (C1(k+1,:)-C1(k,:))/(dz)
   else
       if (k .lt. nlevels-1) then
           Db(k,:) = (B1(k+1,:)-B1(k-1,:))/(4*dz)
           Eb(k,:) = (C1(k+1,:)-C1(k-1,:))/(2*dz)
       else
           Db(k,:) = (B1(k,:)-B1(k-1,:))/(2*dz)
           Eb(k,:) = (C1(k,:)-C1(k-1,:))/(dz)
       end if
   end if 
end do

D = Da + Db
E = Ea + Eb

;=========================================================
; 新增：修复非法值
;=========================================================

A@_FillValue = 9.96921e+36
B@_FillValue = 9.96921e+36
C@_FillValue = 9.96921e+36
D@_FillValue = 9.96921e+36
E@_FillValue = 9.96921e+36
F@_FillValue = 9.96921e+36

n_bad_A = 0
n_bad_B = 0
n_bad_C = 0
n_bad_D = 0
n_bad_E = 0
n_bad_F = 0

do k = 0 , nlevels-1
  do i = 0 , sumlen-1

    ; ---------- A ----------
    if (.not.ismissing(A(k,i))) then
      if (.not.(A(k,i).eq.A(k,i))) then

        if (i .gt. 0) then
          A(k,i) = A(k,i-1)
        else
          if (k .gt. 0) then
            A(k,i) = A(k-1,i)
          else
            A(k,i) = 0.0
          end if
        end if

        n_bad_A = n_bad_A + 1
      end if

    else

      if (i .gt. 0) then
        A(k,i) = A(k,i-1)
      else
        if (k .gt. 0) then
          A(k,i) = A(k-1,i)
        else
          A(k,i) = 0.0
        end if
      end if

      n_bad_A = n_bad_A + 1
    end if


    ; ---------- B ----------
    if (.not.(B(k,i).eq.B(k,i))) then
      if (i .gt. 0) then
        B(k,i) = B(k,i-1)
      else
        if (k .gt. 0) then
          B(k,i) = B(k-1,i)
        else
          B(k,i) = 0.0
        end if
      end if
      n_bad_B = n_bad_B + 1
    end if


    ; ---------- C ----------
    if (.not.(C(k,i).eq.C(k,i))) then
      if (i .gt. 0) then
        C(k,i) = C(k,i-1)
      else
        if (k .gt. 0) then
          C(k,i) = C(k-1,i)
        else
          C(k,i) = 0.0
        end if
      end if
      n_bad_C = n_bad_C + 1
    end if


    ; ---------- D ----------
    if (.not.(D(k,i).eq.D(k,i))) then
      if (i .gt. 0) then
        D(k,i) = D(k,i-1)
      else
        if (k .gt. 0) then
          D(k,i) = D(k-1,i)
        else
          D(k,i) = 0.0
        end if
      end if
      n_bad_D = n_bad_D + 1
    end if


    ; ---------- E ----------
    if (.not.(E(k,i).eq.E(k,i))) then
      if (i .gt. 0) then
        E(k,i) = E(k,i-1)
      else
        if (k .gt. 0) then
          E(k,i) = E(k-1,i)
        else
          E(k,i) = 0.0
        end if
      end if
      n_bad_E = n_bad_E + 1
    end if


    ; ---------- F ----------
    if (.not.(F(k,i).eq.F(k,i))) then
      if (i .gt. 0) then
        F(k,i) = F(k,i-1)
      else
        if (k .gt. 0) then
          F(k,i) = F(k-1,i)
        else
          F(k,i) = 0.0
        end if
      end if
      n_bad_F = n_bad_F + 1
    end if

  end do
end do

;=========================================================
; ⭐ 输出统计信息
;=========================================================
print("==== BAD POINT COUNT ====")
print("A bad = " + n_bad_A)
print("B bad = " + n_bad_B)
print("C bad = " + n_bad_C)
print("D bad = " + n_bad_D)
print("E bad = " + n_bad_E)
print("F bad = " + n_bad_F)

print("==== RANGE ====")
print("A: " + min(A) + " ~ " + max(A))
print("B: " + min(B) + " ~ " + max(B))
print("C: " + min(C) + " ~ " + max(C))
print("D: " + min(D) + " ~ " + max(D))
print("E: " + min(E) + " ~ " + max(E))
print("F: " + min(F) + " ~ " + max(F))

;---- print  Coefficient---
 fbinrecwrite("./SE-A.ieee",0,A)
 fbinrecwrite("./SE-B.ieee",0,B)
 fbinrecwrite("./SE-C.ieee",0,C)
 fbinrecwrite("./SE-D.ieee",0,D)
 fbinrecwrite("./SE-E.ieee",0,E)
 fbinrecwrite("./SE-F.ieee",0,F)
end




