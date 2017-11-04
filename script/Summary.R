#!/usr/bin/env Rscript
setwd("C:/Users/Markus/Sync/Projects/LogicSem/script")
PlotSize <- 1500

Results_ <-
  read.csv(
    file = "Summary_Benchmarks_.csv",
    header = TRUE,
    sep = ",",
    quote = "\"",
    dec = "."
  )

Results <- Results_[order(Results_[, 5]), ]
Benchmark_Width <-
  read.csv(
    file = "Summary_Benchmark_Width_Ready.csv",
    header = TRUE,
    sep = ",",
    quote = "\"",
    dec = "."
  )

Benchmark_Width$X.models.sharpSAT <- as.numeric(Benchmark_Width$X.models.sharpSAT)

Instances_SAT <-
  paste(Benchmark_Width$file_name[Benchmark_Width$sat.unsat == "SATISFIABLE"], ".cnf", sep =
          "")
Instances_UNSAT <-
  paste(Benchmark_Width$file_name[Benchmark_Width$sat.unsat == "UNSATISFIABLE"], ".cnf", sep =
          "")


Instances_Primal_Width_0_5 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_primal_s1234 >= 0 &
                                    Benchmark_Width$width_primal_s1234 <= 5], ".cnf", sep = "")
Instances_Primal_Width_6_14 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_primal_s1234 >= 6 &
                                    Benchmark_Width$width_primal_s1234 <= 14], ".cnf", sep = "")
Instances_Primal_Width_15_30 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_primal_s1234 >= 15 &
                                    Benchmark_Width$width_primal_s1234 <= 30], ".cnf", sep = "")
Instances_Primal_Width_18_30 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_primal_s1234 >= 18 &
                                    Benchmark_Width$width_primal_s1234 <= 30], ".cnf", sep = "")
Instances_Primal_Width_30 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_primal_s1234 <= 30], ".cnf", sep = "")

Instances_Incidence_Width_0_5 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_incidence_s1234 >=
                                    0 &  Benchmark_Width$width_incidence_s1234 <= 5], ".cnf", sep = "")
Instances_Incidence_Width_6_14 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_incidence_s1234 >=
                                    6 &  Benchmark_Width$width_incidence_s1234 <= 14], ".cnf", sep = "")
Instances_Incidence_Width_15_30 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_incidence_s1234 >=
                                    15 &  Benchmark_Width$width_incidence_s1234 <= 30], ".cnf", sep = "")
Instances_Incidence_Width_18_30 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_incidence_s1234 >=
                                    18 &  Benchmark_Width$width_incidence_s1234 <= 30], ".cnf", sep = "")
Instances_Incidence_Width_30 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$width_incidence_s1234 <= 30], ".cnf", sep = "")



Instances_Models_5000 <-
  paste(Benchmark_Width$file_name[Benchmark_Width$X.models.sharpSAT <= 5000], ".cnf", sep = "")

Instances_Models_5000_p <-
  paste(Benchmark_Width$file_name[Benchmark_Width$X.models.sharpSAT > 5000], ".cnf", sep = "")

# create kaktus plot
generateKaktusPlot <-
  function(PlotTitle,
           Instances1,
           Instances2,
           name,
           name1,
           name2) {
    gpusat1_times_ <- Results$time[Instances1]
    gpusat2_times_ <- Results$time[Instances2]
    
    gpusat1_times <-
      as.numeric(sort(gpusat1_times_[gpusat1_times_ < 900]))
    gpusat1_numbers <- as.numeric(1:length(gpusat1_times))
    gpusat2_times <-
      as.numeric(sort(gpusat2_times_[gpusat2_times_ < 900]))
    gpusat2_numbers <- as.numeric(1:length(gpusat2_times))
    
    gpusat_vbest_times <-
      as.numeric(sort(pmin(gpusat1_times_, gpusat2_times_)))
    gpusat_vbest_times <- gpusat_vbest_times[gpusat_vbest_times < 900]
    gpusat_vbest_numbers <- as.numeric(1:length(gpusat_vbest_times))
    
    png(file = name,
        width = 2 * PlotSize,
        height = 1 * PlotSize)
    plot(
      range(0:length(gpusat_vbest_times) + 1),
      range(0:max(gpusat1_times, gpusat2_times)),
      type = "n",
      xlab = "Instances",
      ylab = "Time",
      lab = c(20, 10, 7)
    )
    lines(
      gpusat1_numbers,
      gpusat1_times,
      type = "l",
      lwd = 0.5,
      lty = "solid",
      col = "green",
      pch = "."
    )
    lines(
      gpusat2_numbers,
      gpusat2_times,
      type = "l",
      lwd = 0.5,
      lty = "solid",
      col = "red",
      pch = "."
    )
    lines(
      gpusat_vbest_numbers,
      gpusat_vbest_times,
      type = "l",
      lwd = 0.5,
      lty = "solid",
      col = "blue",
      pch = "."
    )
    title(PlotTitle)
    legend(
      0,
      0,
      xpd = TRUE,
      c(name1, name2, "best"),
      cex = 0.8,
      col = c("green", "red", "blue"),
      lty = "solid",
      title = "solver"
    )
    dev.off()
  }

# create instance plot
generateInstancePlot <-
  function(PlotTitle,
           Instances1,
           Instances2,
           name,
           name1,
           name2) {
    gpusat1_times_ <- Results$time[Instances1]
    gpusat2_times_ <- Results$time[Instances2]
    
    gpusat1_times <- as.numeric((gpusat1_times_))
    gpusat1_numbers <- as.numeric(1:length(gpusat1_times))
    gpusat2_times <- as.numeric((gpusat2_times_))
    gpusat2_numbers <- as.numeric(1:length(gpusat2_times))
    
    gpusat_vbest_times <-
      as.numeric((pmin(gpusat1_times_, gpusat2_times_)))
    gpusat_vbest_numbers <- as.numeric(1:length(gpusat_vbest_times))
    
    gpusat1_times <- gpusat1_times[order(gpusat_vbest_times)]
    gpusat2_times <- gpusat2_times[order(gpusat_vbest_times)]
    gpusat_vbest_times <-
      gpusat_vbest_times[order(gpusat_vbest_times)]
    
    png(file = name,
        width = 2 * PlotSize,
        height = 1 * PlotSize)
    plot(
      range(0:length(gpusat_vbest_times) + 1),
      range(0:max(gpusat1_times, gpusat2_times)),
      type = "n",
      xlab = "Instances",
      ylab = "Time",
      lab = c(20, 10, 7)
    )
    lines(
      gpusat1_numbers,
      gpusat1_times,
      type = "l",
      lwd = 0.5,
      lty = "solid",
      col = "green",
      pch = "."
    )
    lines(
      gpusat2_numbers,
      gpusat2_times,
      type = "l",
      lwd = 0.5,
      lty = "solid",
      col = "red",
      pch = "."
    )
    lines(
      gpusat_vbest_numbers,
      gpusat_vbest_times,
      type = "l",
      lwd = 0.5,
      lty = "solid",
      col = "blue",
      pch = "."
    )
    title(PlotTitle)
    legend(
      0,
      -0.1,
      xpd = TRUE,
      c(name1, name2, "best"),
      cex = 0.8,
      col = c("green", "red", "blue"),
      lty = "solid",
      title = "solver"
    )
    dev.off()
  }

generateInstanceKaktusPlot <-
  function(PlotTitle,
           Instances1,
           Instances2,
           name,
           name1,
           name2) {
    generateKaktusPlot(
      paste("kaktus ", PlotTitle, sep = ""),
      Instances1,
      Instances2,
      paste("./Plots/kaktus_", name, sep = ""),
      name1,
      name2
    )
    generateInstancePlot(
      paste("instance ", PlotTitle, sep = ""),
      Instances1,
      Instances2,
      paste("./Plots/instance_", name, sep = ""),
      name1,
      name2
    )
  }


## gpusat vs gpusat
generateInstanceKaktusPlot(
  "gpusat incidence vs primal",
  Results$setting == "primal_double_w-14",
  Results$setting == "incidence_double_w-14",
  "gpusat_incidence_vs_primal.png",
  "primal",
  "incidence"
)


## gpusat vs. approxmc
generateInstanceKaktusPlot(
  "gpusat primal vs. approxmc",
  Results$setting == "primal_double_w-14",
  Results$solver == "approxmc",
  "gpusat_primal-vs-approxmc.png",
  "gpusat primal",
  "approxmc"
)
generateInstanceKaktusPlot(
  "gpusat incidence vs. approxmc",
  Results$setting == "incidence_double_w-14",
  Results$solver == "approxmc",
  "gpusat_incidence-vs-approxmc.png",
  "gpusat incidence",
  "approxmc"
)


## gpusat (vbest) vs. approxmc
### primal
### incidence


## gpusat vs. sharpSAT
generateInstanceKaktusPlot(
  "gpusat primal vs. sharpSAT",
  Results$setting == "primal_double_w-14",
  Results$solver == "sharpSAT",
  "gpusat_primal-vs-sharpSAT.png",
  "gpusat primal",
  "sharpSAT"
)
generateInstanceKaktusPlot(
  "gpusat incidence vs. sharpSAT",
  Results$setting == "incidence_double_w-14",
  Results$solver == "sharpSAT",
  "gpusat_incidence-vs-sharpSAT.png",
  "gpusat incidence",
  "sharpSAT"
)


## gpusat vs. cachet
generateInstanceKaktusPlot(
  "gpusat primal vs. cachet",
  Results$setting == "primal_double_w-14",
  Results$solver == "cachet",
  "gpusat_primal-vs-cachet.png",
  "gpusat primal",
  "cachet"
)
generateInstanceKaktusPlot(
  "gpusat incidence vs. cachet",
  Results$setting == "incidence_double_w-14",
  Results$solver == "cachet",
  "gpusat_incidence-vs-cachet.png",
  "gpusat incidence",
  "cachet"
)


## gpusat vs. dsharp
generateInstanceKaktusPlot(
  "gpusat primal vs. dsharp",
  Results$setting == "primal_double_w-14",
  Results$solver == "dsharp",
  "gpusat_primal-vs-dsharp.png",
  "gpusat primal",
  "dsharp"
)
generateInstanceKaktusPlot(
  "gpusat incidence vs. dsharp",
  Results$setting == "incidence_double_w-14",
  Results$solver == "dsharp",
  "gpusat_incidence-vs-dsharp.png",
  "gpusat incidence",
  "dsharp"
)


## SAT gpusat vs. dynasp
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal SAT",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_SAT,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_SAT,
  "gpusat_primal-vs-dynasp-SAT.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence SAT",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_SAT,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_SAT,
  "gpusat_incidence-vs-dynasp-SAT.png",
  "gpusat incidence",
  "dynasp incidence"
)


## UNSAT gpusat vs. dynasp
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal UNSAT",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_UNSAT,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_UNSAT,
  "gpusat_primal-vs-dynasp-UNSAT.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence UNSAT",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_UNSAT,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_UNSAT,
  "gpusat_incidence-vs-dynasp-UNSAT.png",
  "gpusat incidence",
  "dynasp incidence"
)


## gpusat vs. dynasp: Width 0-5
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal Width 0-5",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_Primal_Width_0_5,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_Primal_Width_0_5,
  "gpusat_primal-vs-dynasp-Width_0-5.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence Width 0-5",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_Incidence_Width_0_5,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_Incidence_Width_0_5,
  "gpusat_incidence-vs-dynasp-Width_0-5.png",
  "gpusat incidence",
  "dynasp incidence"
)


## gpusat vs. dynasp: Width 6-14
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal Width 6-14",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_Primal_Width_6_14,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_Primal_Width_6_14,
  "gpusat_primal-vs-dynasp-Width_6-14.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence Width 6-14",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_Incidence_Width_6_14,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_Incidence_Width_6_14,
  "gpusat_incidence-vs-dynasp-Width_6-14.png",
  "gpusat incidence",
  "dynasp incidence"
)


## gpusat vs. dynasp: Width 15-30
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal Width 15-30",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_Primal_Width_15_30,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_Primal_Width_15_30,
  "gpusat_primal-vs-dynasp-Width_15-30.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence Width 15-30",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_Incidence_Width_15_30,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_Incidence_Width_15_30,
  "gpusat_incidence-vs-dynasp-Width_15-30.png",
  "gpusat incidence",
  "dynasp incidence"
)


## gpusat vs. dynasp: Width 18-30
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal Width 18-30",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_Primal_Width_18_30,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_Primal_Width_18_30,
  "gpusat_primal-vs-dynasp-Width_18-30.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence Width 18-30",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_Incidence_Width_18_30,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_Incidence_Width_18_30,
  "gpusat_incidence-vs-dynasp-Width_18-30.png",
  "gpusat incidence",
  "dynasp incidence"
)


## gpusat vs. dynasp: Width <30
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal Width 30",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_Primal_Width_30,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_Primal_Width_30,
  "gpusat_primal-vs-dynasp-Width_30.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence Width 30",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_Incidence_Width_30,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_Incidence_Width_30,
  "gpusat_incidence-vs-dynasp-Width_30.png",
  "gpusat incidence",
  "dynasp incidence"
)


## dynasp vs. sharpsat
generateInstanceKaktusPlot(
  "dynasp vs. sharpsat",
  Results$setting == "dynasp_primal"&
    Results$instance %in% Instances_Primal_Width_30,
  Results$solver == "sharpSAT"&
    Results$instance %in% Instances_Primal_Width_30,
  "dynasp_primal-vs-sharpsat.png",
  "dynasp primal",
  "sharpsat"
)

generateInstanceKaktusPlot(
  "dynasp vs. sharpsat",
  Results$setting == "dynasp_incidence"&
    Results$instance %in% Instances_Incidence_Width_30,
  Results$solver == "sharpSAT"&
    Results$instance %in% Instances_Incidence_Width_30,
  "dynasp_incidence-vs-sharpsat.png",
  "dynasp incidence",
  "sharpsat"
)


## gpusat vs. dynasp: Solutions less than 5000
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal models <= 5000",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_Primal_Width_30 & Results$instance %in% Instances_Models_5000,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_Primal_Width_30 & Results$instance %in% Instances_Models_5000,
  "gpusat_primal-vs-dynasp-models_l_5000.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence models <= 5000",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_Incidence_Width_30 & Results$instance %in% Instances_Models_5000,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_Incidence_Width_30 & Results$instance %in% Instances_Models_5000,
  "gpusat_incidence-vs-dynasp-models_l_5000.png",
  "gpusat incidence",
  "dynasp incidence"
)


## gpusat vs. dynasp: Solutions more than 5000
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal models > 5000",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_Primal_Width_30 & Results$instance %in% Instances_Models_5000_p,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_Primal_Width_30 & Results$instance %in% Instances_Models_5000_p,
  "gpusat_primal-vs-dynasp-models_g_5000.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence models > 5000",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_Incidence_Width_30 & Results$instance %in% Instances_Models_5000_p,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_Incidence_Width_30 & Results$instance %in% Instances_Models_5000_p,
  "gpusat_incidence-vs-dynasp-models_g_5000.png",
  "gpusat incidence",
  "dynasp incidence"
)


## gpusat vs. dynasp: Solutions more than 5000
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal models > 5000 width 18-30",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_Primal_Width_18_30 & Results$instance %in% Instances_Models_5000_p,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_Primal_Width_18_30 & Results$instance %in% Instances_Models_5000_p,
  "gpusat_primal-vs-dynasp-Width_18-30-models_g_5000.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence models > 5000 width 18-30",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_Incidence_Width_18_30 & Results$instance %in% Instances_Models_5000_p,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_Incidence_Width_18_30 & Results$instance %in% Instances_Models_5000_p,
  "gpusat_incidence-vs-dynasp-Width_18-30-models_g_5000.png",
  "gpusat incidence",
  "dynasp incidence"
)


## gpusat vs. dynasp: Solutions more than 5000
generateInstanceKaktusPlot(
  "gpusat vs. dynasp primal models <= 5000 width 18-30",
  Results$setting == "primal_double_w-14" &
    Results$instance %in% Instances_Primal_Width_18_30 & Results$instance %in% Instances_Models_5000,
  Results$setting == "dynasp_primal" &
    Results$instance %in% Instances_Primal_Width_18_30 & Results$instance %in% Instances_Models_5000,
  "gpusat_primal-vs-dynasp-Width_18-30-models_l_5000.png",
  "gpusat primal",
  "dynasp primal"
)

generateInstanceKaktusPlot(
  "gpusat vs. dynasp incidence models <= 5000 width 18-30",
  Results$setting == "incidence_double_w-14" &
    Results$instance %in% Instances_Incidence_Width_18_30 & Results$instance %in% Instances_Models_5000,
  Results$setting == "dynasp_incidence" &
    Results$instance %in% Instances_Incidence_Width_18_30 & Results$instance %in% Instances_Models_5000,
  "gpusat_incidence-vs-dynasp-Width_18-30-models_l_5000.png",
  "gpusat incidence",
  "dynasp incidence"
)
