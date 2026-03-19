// HOVER ACTIVE ICON ON SIDEBAR 
const sections = document.querySelectorAll("section");
const navLinks = document.querySelectorAll(".menu-item");

window.addEventListener("scroll", () => {

let current = "";

sections.forEach(section => {
const sectionTop = section.offsetTop;
const sectionHeight = section.clientHeight;

if (pageYOffset >= sectionTop - 200) {
current = section.getAttribute("id");
}
});

navLinks.forEach(link => {
link.classList.remove("active");

if(link.getAttribute("href") === "#" + current){
link.classList.add("active");
}
});

});

// STAR RATING DISTRIBUTION
new Chart(document.getElementById("ratingChart"), {
  type: "bar",
  data: {
    labels: ["1", "2", "3", "4", "5"],
    datasets: [{
      label: "Number of Reviews",
      data: [5210000, 8370000, 9030000, 17680000, 62610000],
      backgroundColor: "#336193",
      borderRadius: 5,
      barPercentage: 0.8
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          // Format Y-axis labels
          callback: function(value) {
            return (value / 1000000) + "M"; // convert to millions
          }
        }
      }
    }
  }
});


// TRAIN AND TEST ERRORS LINE PLOT
const ctxerrors = document.getElementById("traintesterrors").getContext("2d");

// Number of trees from your table
const numTrees = [
  1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
  21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
  41, 43, 45, 47, 49
];

// Train and Test Accuracy values from your table
const trainAccuracy = [
  0.665744, 0.654113, 0.659351, 0.664171, 0.663712,
  0.662680, 0.665152, 0.664260, 0.665103, 0.665590,
  0.666150, 0.665840, 0.665378, 0.666399, 0.666117,
  0.665530, 0.665942, 0.666640, 0.666675, 0.666259,
  0.665892, 0.666353, 0.666352, 0.665896, 0.666061
];

const testAccuracy = [
  0.667175, 0.655575, 0.660583, 0.665423, 0.665066,
  0.664242, 0.666928, 0.665780, 0.666773, 0.667298,
  0.667992, 0.667460, 0.667019, 0.667947, 0.667661,
  0.667253, 0.667908, 0.668167, 0.668265, 0.667811,
  0.667720, 0.668090, 0.668129, 0.667408, 0.667382
];

// Map to Chart.js {x, y} format
const trainData = numTrees.map((x, i) => ({x: x, y: trainAccuracy[i]}));
const testData  = numTrees.map((x, i) => ({x: x, y: testAccuracy[i]}));

new Chart(ctxerrors, {
    type: "line",
    data: {
        datasets: [
            {
                label: "Train Accuracy",
                data: trainData,
                borderColor: "rgb(127, 187, 252)",
                backgroundColor: "rgb(127, 187, 252)",
                fill: false,
                tension: 0.3,
                pointStyle: "circle",
                pointRadius: 5
            },
            {
                label: "Validation Accuracy",
                data: testData,
                borderColor: "rgb(255, 183, 0)",
                backgroundColor: "rgb(255, 183, 0)",
                fill: false,
                tension: 0.3,
                pointStyle: "rect",
                pointRadius: 5
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                display: true,
                position: "top",
                labels: {
                    color: "black",
                    usePointStyle: true,
                    pointStyle: "rect",
                    boxWidth: 20,
                    padding: 10
                }
            }
        },
        scales: {
            x: {
                type: "linear",
                min: 0,
                max: 50,
                title: {
                    display: true,
                    text: "Number of Trees",
                    color: "black",
                    font: { size: 16, weight: "bold" }
                },
                ticks: { color: "black", stepSize: 10 }
            },
            y: {
                min: 0.65,
                max: 0.67,
                title: {
                    display: true,
                    text: "Accuracy",
                    color: "black",
                    font: { size: 16, weight: "bold" }
                },
                ticks: { color: "black" }
            }
        }
    }
});

// RANDOM FOREST FEATURE IMPORTANCE BAR CHART
const features = [
    'review_word_counts',
    'review_len',
    'helpful_ratio',
    'review_headline_len',
    'category_idx',
    'total_votes',
    'review_headline_word_counts',
    'verified_purchase_idx'
];

const importances = [
    0.248081,
    0.210397,
    0.169301,
    0.126052,
    0.122743,
    0.105226,
    0.018200,
    0.000000
];

const ctximportance = document.getElementById('featureimportance').getContext('2d');

new Chart(ctximportance, {
    type: 'bar',
    data: {
        labels: features,
        datasets: [{
            label: 'Importance',
            data: importances,
            backgroundColor: 'rgba(0, 100, 207, 0.7)',
            borderColor: 'rgba(0, 100, 207, 0.7)',
            borderWidth: 1
        }]
    },
    options: {
        indexAxis: 'y', // horizontal bar chart
        responsive: true,
        plugins: {
            legend: { display: false }, // optional
            tooltip: {
                enabled: true,
                callbacks: {
                    label: function(context) {
                        return context.raw.toFixed(5); // show 5 decimals
                    }
                }
            }
        },
        scales: {
            x: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Importance',
                    font: { size: 16, weight: 'bold' }
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Features',
                    font: { size: 16, weight: 'bold' }
                },
                ticks: { color: 'black' }
            }
        }
    }
});

// MODEL PERFORMANCE CHART
const ctx = document.getElementById('modelPerformanceChart').getContext('2d');

const modelPerformanceChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: [
            'LogReg',
            'RF A',
            'RF B',
            'RF C'
        ],
        datasets: [
            {
                label: 'Train Accuracy',
                data: [0.6084, 0.6098, 0.6085, 0.6119],
                backgroundColor: 'rgba(59, 144, 200, 0.7)'
            },
            {
                label: 'Validation Accuracy',
                data: [0.6069, 0.6083, 0.6070, 0.6103],
                backgroundColor: 'rgba(255, 117, 99, 0.7)'
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            tooltip: {
                enabled: true,
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1, // accuracy is between 0 and 1
                ticks: {
                    callback: function(value) { return (value * 100).toFixed(0) + '%'; }
                },
                title: {
                    display: true,
                    text: 'Accuracy'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Model'
                }
            }
        }
    }
});

// MACHINE LEARNING 
const ctxValRMSE = document.getElementById('valRMSEChart').getContext('2d');
const valRMSEChart = new Chart(ctxValRMSE, {
    type: 'bar',
    data: {
        labels: ['LogReg', 'RF_A', 'RF_B', 'RF_C'],
        datasets: [{
            label: 'Validation RMSE',
            data: [1.53, 1.52, 1.52, 1.51], // Replace with actual numbers from Python
            backgroundColor: '#3B82F6'
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: { display: false } },
        title: { display: true, text: 'Validation RMSE Comparison' }
    }
});