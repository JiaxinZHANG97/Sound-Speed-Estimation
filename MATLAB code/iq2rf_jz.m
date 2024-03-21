function rf = iq2rf_jz(I0,Q0,f0,fs,decim,intf,centerAngle_flag)

if  centerAngle_flag
    [m,n]=size(I0);
    rf = zeros(intf*m,n);
    sample_time_interval = decim/(fs*intf);
    t = 0:sample_time_interval:(intf*m-1)*sample_time_interval;
    theta = 2*pi*f0*t;
    c = cos(theta); c=c(:);
    s = sin(theta); s=s(:);
    
    for aline=1:n
        rf(:,aline)=interpft(I0(:,aline),m*intf).*c - interpft(Q0(:,aline),m*intf).*s;
    end

else
    [m,n,pw]=size(I0);
    rf = zeros(intf*m,n,pw);
    sample_time_interval = decim/(fs*intf);
    t = 0:sample_time_interval:(intf*m-1)*sample_time_interval;
    theta = 2*pi*f0*t;
    c = cos(theta); c=c(:);
    s = sin(theta); s=s(:);
    
    for angle = 1:pw
        for aline=1:n
            rf(:,aline,angle)=interpft(I0(:,aline,angle),m*intf).*c - interpft(Q0(:,aline,angle),m*intf).*s;
        end
    end
end

