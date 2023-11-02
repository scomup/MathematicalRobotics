# Camera reprojection.  

The reprojection errors $r$ indicate the difference between the projected image point $u_{c_j}^*$ and the observed image point $u_{c_j}$.

$$ 
r = u_{c_j}^*(p_{c_j}(T_{c_j,c_i}(T_{w,b_i},T_{w,b_j}),p_{c_i}(d))) - u_{c_j} 
\tag{1}
$$

$$
u_{c_j}^* = K^{-1}p_{c_j}
\tag{2}
$$

$$
p_{c_j} = T_{c_j,c_i}p_{c_i}
\tag{3}
$$

$$
T_{c_j,c_i} = T_{bc}^{-1}T_{w,b_j}^{-1}T_{w,b_i}T_{bc}
\tag{4}
$$

* K: camera intrinsic matrix
* $u_{c_j}^*$: The projected image point in c2 from c1
* $u_{c_j}$: The observed image in c2
* $b$: The body frame
* $c_i$: The camera1 frame
* $c_j$: The camera2 frame
* $w$: The world frame
* $T$: Transform matrix
* $p$: 3d point
* $d$: The depth of the point in camera1

### Jocabian of r


$$ 
J_{T_{w,c1}}^r = J_{p_{c_j}}^{u_{c_j}^*} J_{T_{c_j,c_i}}^{p_{c_j}} J_{T_{w,b_i}}^{T_{c_j,c_i}}
\tag{5}
$$

$$ 
J_{T_{w,c2}}^r = J_{p_{c_j}}^{u_{c_j}^*} J_{T_{c_j,c_i}}^{p_{c_j}} J_{T_{w,b_j}}^{T_{c_j,c_i}}
\tag{6}
$$

$$ 
J_{d}^r = J_{p_{c_j}}^{u_{c_j}^*} J_{p_{c_i}}^{p_{c_j}} J_{d}^{p_{c_i}}
\tag{7}
$$

### Jocabian of $T_{w,b_i}$ for $T_{c_j,c_i}$

$$
T_{c_j,c_i} = T_{bc}^{-1}T_{w,b_j}^{-1}T_{w,b_i}T_{bc}
$$

$$
J_{T_{w,b_i}}^{T_{c_j,c_i}} = -T_{bc}^{-1} T_{w,b_i}^{-1}T_{w,b_j}
$$