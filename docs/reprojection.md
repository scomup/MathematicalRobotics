# Camera reprojection.  

The reprojection errors $r$ indicate the difference between the projected image point $u_{c_2}^*$ and the observed image point $u_{c_2}$.
$$ 
r = u_{c_2}^*(p_{c_2}(T_{c_2,c_1}(T_{w,b_1},T_{w,b_2}),p_{c_1}(d))) - u_{c_2} 
\tag{1}
$$

$$
u_{c_2}^* = K^{-1}p_{c_2}
\tag{2}
$$

$$
p_{c_2} = T_{c_2,c_1}p_{c_1}
\tag{3}
$$

$$
T_{c_2,c_1} = T_{bc}^{-1}T_{w,b_2}^{-1}T_{w,b_1}T_{bc}
\tag{4}
$$

* K: camera intrinsic matrix
* $u_{c_2}^*$: The projected image point in c2 from c1
* $u_{c_2}$: The observed image in c2
* $b$: The body frame
* $c_1$: The camera1 frame
* $c_2$: The camera2 frame
* $w$: The world frame
* $T$: Transform matrix
* $p$: 3d point
* $d$: The depth of the point in camera1

### Jocabian of r


$$ 
J_{T_{w,c1}}^r = J_{p_{c_2}}^{u_{c_2}^*} J_{T_{c_{2},c_{1}}}^{p_{c_2}} J_{T_{w,b_1}}^{T_{c_{2},c_{1}}}
\tag{5}
$$

$$ 
J_{T_{w,c2}}^r = J_{p_{c_2}}^{u_{c_2}^*} J_{T_{c_{2},c_{1}}}^{p_{c_2}} J_{T_{w,b_2}}^{T_{c_{2},c_{1}}}
\tag{6}
$$

$$ 
J_{d}^r = J_{p_{c_2}}^{u_{c_2}^*} J_{p_{c_1}}^{p_{c_2}} J_{d}^{p_{c_1}}
\tag{7}
$$

### Jocabian of $T_{w,b_1}$ for $T_{c_{2},c_{1}}$

$$
J_{T_{w,b_1}}^{T_{c_{2},c_{1}}} = \\
s
$$