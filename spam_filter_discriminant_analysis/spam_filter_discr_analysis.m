function spam_filter_discr_analysis()
    

    a = load('spamdata.mat'); %show the struct in spamdata.mat
    
    X = a.training_set;
    y = a.training_set_label;
    m = length(a.training_set_label);
    
    Xout = a.testing_set;
    yout = a.testing_set_label;
    mout = length(a.testing_set_label);
    
    phi = (1/m)*sum(y); %probability that y is 1
    %phi roughly .4
    
    mu0 = zeros(1,48);
    mu1 = zeros(1,48);
    
    for j = 1:48
        mu0(j) = (1/(m - sum(y))) * sum((X(:,j)).*(1-y)); %average values of X cols where y=0
        mu1(j) = (1/sum(y)) * sum((X(:,j)).*y); %average values of X cols where y=0
    end
    
    cov2 = zeros(48);
    for i = 1:m
        if y(i) == 0
            cov2 = cov2 + ((X(i,:) - mu0)*transpose(X(i,:) - mu0));
        elseif y(i) == 1
            cov2 = cov2 + ((X(i,:) - mu1)*transpose(X(i,:) - mu1));
        end
    end
    %single covariance matrix for GDA
    
    cov2 = cov2 .* (1/m);    
    cov2 = cov2 + 0.01*eye(48); %matrix is rank deficient without adding
    
    %%%%%%% GDA training set
    
    %check which mean generates a higher probability (mean for 0 or 1), and
    %select element whose mean was closer (i.e 0 or 1)
    predictionsDec = zeros(m, 1);
    for o = 1:m
        pred0 = exp( (-1/2) * (X(o,:) - mu0) * inv(cov2) * transpose(X(o,:) - mu0) ) * (1/ ((2*pi)^(48/2) * det(cov2)^(1/2)));
        pred1 = exp( (-1/2) * (X(o,:) - mu1) * inv(cov2) * transpose(X(o,:) - mu1) ) * (1/ ((2*pi)^(48/2) * det(cov2)^(1/2)));
        if pred1 > pred0
            predictionsDec(o,1) = 1;
        end
    end
    
    %combine output of predictions with actual output (y) for ease of
    %comparison
    both = [transpose(predictionsDec); transpose(y)];
    
    %find how many correct
    correct = 0;
    for p = 1:m
        if sum(both(:,p)) == 0 || sum(both(:,p)) == 2
            correct = correct + 1;
        end
    end
    
    disp('GDA training sample error is:')
    disp(100 - 100 * (correct/m)) %find percentage correct out of total values
    
    %%%%%% GDA testing_set
    
    %as above, but replace X with Xout (as testing on new data) and y with
    %yout, as well as changing dimensions from size m (3601) to mout (1000)
    predictionsDecOut = zeros(mout, 1);
    for o2 = 1:mout
        pred0 = exp( (-1/2) * (Xout(o2,:) - mu0) * inv(cov2) * transpose(Xout(o2,:) - mu0) ) * (1/ ((2*pi)^(48/2) * det(cov2)^(1/2)));
        pred1 = exp( (-1/2) * (Xout(o2,:) - mu1) * inv(cov2) * transpose(Xout(o2,:) - mu1) ) * (1/ ((2*pi)^(48/2) * det(cov2)^(1/2)));
        if pred1 > pred0
            predictionsDecOut(o2,1) = 1;
        end
    end
    
    bothOut = [transpose(predictionsDecOut); transpose(yout)];
    
    correctOut = 0;
    for p2 = 1:mout
        if sum(bothOut(:,p2)) == 0 || sum(bothOut(:,p2)) == 2
            correctOut = correctOut + 1;
        end
    end
    
    disp('GDA test sample error is:')
    disp(100 - 100 * (correctOut/mout)) %logical that test sample would have more error since the data is trained on the training data
    
    
    %%%%%%%%%%%%%% NB GDA in sample
    
    %probability that y=1 given x 
    %product of p(xi|y=0) 
    
    %single covariance matrix - covariance of each feature multiplied by
    %covariance of each feature
    covNBGDA = eye(48);
    for u = 1:48
        covNBGDA(u,u) = var(X(:,u));
    end
     
    insampleResults = zeros(3601,1);
    
    for b = 1:3601
        
        pxiy0 = (1-phi); %multiply all probabilities together, initialize to p(y=0)
        pxiy1 = phi; %same, initialize to p(y=1)
        
        for k = 1:48
            
            %multiply by a constant for both so that number doesn't become
            %too small. it isnt necessary to divide by p(x) since i'm only
            %comparing them to each other
            p0 = (1e+20) * (exp( (-1/2) * (X(b,k) - mu0(k)) * cov2(k,k) * (X(b,k) - mu0(k)) ) * (1/ ((2*pi)^(48/2) * (cov2(k,k)^(1/2)))));
            pxiy0 = pxiy0*p0;
        
            p1 = (1e+20) * (exp( (-1/2) * (X(b,k) - mu1(k)) * cov2(k,k) * (X(b,k) - mu1(k)) ) * (1/ ((2*pi)^(48/2) * (cov2(k,k)^(1/2)))));
            pxiy1 = pxiy1*p1;
                    
        end
        
        %pick which function (centered around mean0 or mean1) returned a
        %higher probability (won't add to 1 since didn't divide by p(x))
        %and select that value (0 or 1)
        if pxiy1 > pxiy0
            insampleResults(b,1) = 1; %else 0 already
        end
        
    end
    
    bothNBGDAinsample = [transpose(insampleResults); transpose(y)];
        
    correctNBGDAinsample = 0;
    for p3 = 1:m
        if sum(bothNBGDAinsample(:,p3)) == 0 || sum(bothNBGDAinsample(:,p3)) == 2
            correctNBGDAinsample = correctNBGDAinsample + 1;
        end
    end
    
    disp('NBGDA training sample error is:')
    disp(100 - 100 * (correctNBGDAinsample/m))
    
    
    %%%%%%%%% NBGDA out of sample
    %as above, but different data
    
    outofsampleResults = zeros(1000,1);
    
    for b = 1:1000
        
        pxiy0 = (1-phi); %multiply all probabilities together, initialize to p(y=0)
        pxiy1 = phi; %same, initialize to p(y=1)
        
        for k = 1:48
        
            p0 = (1e+20) * (exp( (-1/2) * (Xout(b,k) - mu0(k)) * cov2(k,k) * (Xout(b,k) - mu0(k)) ) * (1/ ((2*pi)^(48/2) * (cov2(k,k)^(1/2)))));
            pxiy0 = pxiy0*p0;
        
            p1 = (1e+20) * (exp( (-1/2) * (Xout(b,k) - mu1(k)) * cov2(k,k) * (Xout(b,k) - mu1(k)) ) * (1/ ((2*pi)^(48/2) * (cov2(k,k)^(1/2)))));
            pxiy1 = pxiy1*p1;
                    
        end
        
        if pxiy1 > pxiy0
            outofsampleResults(b,1) = 1; %else 0 already
        end
        
    end
    
    bothNBGDAoutsample = [transpose(outofsampleResults); transpose(yout)];
        
    correctNBGDAoutofsample = 0;
    for p3 = 1:1000
        if sum(bothNBGDAoutsample(:,p3)) == 0 || sum(bothNBGDAoutsample(:,p3)) == 2
            correctNBGDAoutofsample = correctNBGDAoutofsample + 1;
        end
    end
    
    disp('NBGDA test sample error is:')
    disp(100 - 100 * (correctNBGDAoutofsample/1000))
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NB BDA
    
    
    BA_X = zeros(3601,48);
    %set all nonzero values to 1
    BA_X(X > 0) = 1;
    
    numberOfY0 = length(y) - sum(y);
    numberOfY1 = sum(y);
    
    probYeq0 = numberOfY0 / length(y);
    probYeq1 = numberOfY1 / length(y);
    
    fullProbabilities = zeros(4,48);
    % 4 rows of bernoulli probabilities
    % 1st row - p(x=0|y=0)
    % 2nd row - p(x=1|y=0)
    % 3rd row - p(x=0|y=1)
    % 4th row - p(x=1|y=1)
    
    % p(x=k, y=c) = number of k's in column of x / number of c's in column of y
    % p(x=k|y=c) = p(x=k, y=c) / p(y=c)
    
    
    for z = 1:48
        
        numberOf0sInX0sInY = sum((1 - BA_X(:,z)) .* (1-y));
        numberOf1sInX0sInY = sum((BA_X(:,z)) .* (1-y));
        numberOf0sInX1sInY = sum((1 - BA_X(:,z)) .* y);
        numberOf1sInX1sInY = sum(BA_X(:,z) .* y);
        
        numberOf0sInY = length(y) - sum(y);
        numberOf1sInY = sum(y);
        
        px0y0 = numberOf0sInX0sInY / numberOf0sInY;
        px1y0 = numberOf1sInX0sInY / numberOf0sInY;
        px0y1 = numberOf0sInX1sInY / numberOf1sInY;
        px1y1 = numberOf1sInX1sInY / numberOf1sInY;
        
        fullProbabilities(1,z) = px0y0;
        fullProbabilities(2,z) = px1y0;
        fullProbabilities(3,z) = px0y1;
        fullProbabilities(4,z) = px1y1;
        
    end
    
    %NBBDA in sample
    
    predictionNBBDA = zeros(3601,1);
   
   for g = 1:m
       
        %initialize p(y|x) 
        pyx0 = probYeq0;
        pyx1 = probYeq1;
        
        
        for f = 1:48
            %y=0
            if X(g,f) == 0
                pyx0 = pyx0 * fullProbabilities(1,f);
            elseif X(g,f) == 1
                pyx0 = pyx0 * fullProbabilities(2,f);
            end
            
            %y=1
            if X(g,f) == 0
                pyx1 = pyx1 * fullProbabilities(3,f);
            elseif X(g,f) == 1
                pyx1 = pyx1 * fullProbabilities(4,f);
            end
            
        end
        
        if pyx1 > pyx0
            predictionNBBDA(g,1) = 1;
        end 
   end
   
   bothNBBDA = [transpose(predictionNBBDA); transpose(y)];
        
   correctNBBDA = 0;
   for p4 = 1:m
       if sum(bothNBBDA(:,p4)) == 0 || sum(bothNBBDA(:,p4)) == 2
           correctNBBDA = correctNBBDA + 1;
       end
   end
    
   disp('NBBDA training sample error is:');
   disp(100 - 100 * (correctNBBDA/m));
   
   
   
   %NBBDA out of sample
    
   predictionNBBDAout = zeros(1000,1);
   
   for g = 1:1000
       
        %initialize p(y|x) 
        pyx0 = probYeq0;
        pyx1 = probYeq1;
        
        
        for f = 1:48
            %y=0
            if Xout(g,f) == 0
                pyx0 = pyx0 * fullProbabilities(1,f);
            elseif Xout(g,f) == 1
                pyx0 = pyx0 * fullProbabilities(2,f);
            end
            
            %y=1
            if Xout(g,f) == 0
                pyx1 = pyx1 * fullProbabilities(3,f);
            elseif Xout(g,f) == 1
                pyx1 = pyx1 * fullProbabilities(4,f);
            end
            
        end
        
        if pyx1 > pyx0
            predictionNBBDAout(g,1) = 1;
        end 
   end
   
   bothNBBDAout = [transpose(predictionNBBDAout); transpose(yout)];
        
   correctNBBDAout = 0;
   for p5 = 1:1000
       if sum(bothNBBDAout(:,p5)) == 0 || sum(bothNBBDAout(:,p5)) == 2
           correctNBBDAout = correctNBBDAout + 1;
       end
   end
    
   disp('NBBDA test sample error is:');
   disp(100 - 100 * (correctNBBDAout/1000));
   
   
   %%%%%%%%%% QDA training set
   
   %separate covariance matrix for spam and not spam
   covQDA0 = zeros(48);
   covQDA1 = zeros(48);
    for i = 1:m
        if y(i) == 0
            %covariance matrix of 0 only considers elements where y is 0
            covQDA0 = covQDA0 + ((X(i,:) - mu0)*transpose(X(i,:) - mu0));
        elseif y(i) == 1
            %as above, but with 1
            covQDA1 = covQDA1 + ((X(i,:) - mu1)*transpose(X(i,:) - mu1));
        end
    end
    
    covQDA0 = covQDA0 .* (1/m);    
    covQDA0 = covQDA0 + 0.01*eye(48); %matrix is rank deficient without adding
    
    covQDA1 = covQDA1 .* (1/m);    
    covQDA1 = covQDA1 + 0.01*eye(48); %matrix is rank deficient without adding
    
    
    predictionsQDAin = zeros(m, 1);
    for o = 1:m
        %same principle as for GDA, but with different means AND covariance
        pred0 = exp( (-1/2) * (X(o,:) - mu0) * inv(covQDA0) * transpose(X(o,:) - mu0) ) * (1/ ((2*pi)^(48/2) * det(covQDA0)^(1/2)));
        pred1 = exp( (-1/2) * (X(o,:) - mu1) * inv(covQDA1) * transpose(X(o,:) - mu1) ) * (1/ ((2*pi)^(48/2) * det(covQDA1)^(1/2)));
        if pred1 > pred0
            predictionsQDAin(o,1) = 1;
        end
    end
    
    bothQDAin = [transpose(predictionsQDAin); transpose(y)];
    
    correctQDAin = 0;
    for p = 1:m
        if sum(bothQDAin(:,p)) == 0 || sum(bothQDAin(:,p)) == 2
            correctQDAin = correctQDAin + 1;
        end
    end
    
    disp('QDA training sample error is:')
    disp(100 - 100 * (correctQDAin/m))
   
   
    
   %%%%%%%%%% QDA testing set
 
    
    predictionsQDAout = zeros(1000, 1);
    for o = 1:1000
        pred0 = exp( (-1/2) * (Xout(o,:) - mu0) * inv(covQDA0) * transpose(Xout(o,:) - mu0) ) * (1/ ((2*pi)^(48/2) * det(covQDA0)^(1/2)));
        pred1 = exp( (-1/2) * (Xout(o,:) - mu1) * inv(covQDA1) * transpose(Xout(o,:) - mu1) ) * (1/ ((2*pi)^(48/2) * det(covQDA1)^(1/2)));
        if pred1 > pred0
            predictionsQDAout(o,1) = 1;
        end
    end
    
    bothQDAout = [transpose(predictionsQDAout); transpose(yout)];
    
    correctQDAout = 0;
    for p = 1:1000
        if sum(bothQDAout(:,p)) == 0 || sum(bothQDAout(:,p)) == 2
            correctQDAout = correctQDAout + 1;
        end
    end
    
    disp('QDA test sample error is:')
    disp(100 - 100 * (correctQDAout/1000))
 
   
end

