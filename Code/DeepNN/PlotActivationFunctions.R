# Setting the book's color scheme and figure settings
library("RColorBrewer")
plotColors = brewer.pal(12, "Paired")
pointColor = plotColors[5] # Color for single dots
lwdDef = 8                 # Default line thickness
lwdThin = 6
lwdThinner = 3
pointSizeDef = 4
cexLabDef = 1.5            # Default scaling of font size labels
cexAxisDef = 1.5           # Default scaling of tick labels

# Disable scientific notation in the plots
options(scipen = 999)

xGrid = seq(-3,3,length = 1000)
ReLU = apply(cbind(0,xGrid), 1, FUN = max)
postscript('ActivationFunctions.eps')
plot(xGrid, 1/(1+exp(-xGrid)), type = "l", axes=FALSE, xlab = "x", 
          ylab = "Activation", cex.lab=cexLabDef, cex.axis = cexAxisDef, 
          col = plotColors[2], lwd = lwdThin, main = "", ylim = c(-1, 3), 
          xlim = c(-3,3)) 
lines(xGrid, ReLU, col = plotColors[4], lwd = lwdThin)
lines(xGrid, tanh(xGrid), col = plotColors[6], lwd = lwdThin)
lines(xGrid, log(1+exp(xGrid)), col = plotColors[7], lwd = lwdThinner)
LeakyReLU = ReLU
LeakyReLU[ReLU<=0]=0.02*xGrid[ReLU<=0]
lines(xGrid, LeakyReLU, col = plotColors[8], lwd = lwdThinner)

axis(side = 1, at = seq(-3,3, by = 1))
axis(side = 2, at = seq(-1,3, by = 1))
legend(x = "topleft", inset=.05, legend = c("Logistic", "Rectified Linear (ReLU)", "tanh","softplus","Leaky ReLU, 0.02"), 
       cex = 1, col = c(plotColors[2], plotColors[4],plotColors[6], plotColors[7], plotColors[8]), 
       lwd = c(lwdThin,lwdThin,lwdThin,lwdThinner,lwdThinner))
dev.off()
 

