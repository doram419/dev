// Chart.js를 사용하여 그래프 초기화
var ctx = document.getElementById('myChart').getContext('2d');
var myChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['1', '2', '3', '4', '5'],
        datasets: [{
            label: '파란 선',
            data: [0, 1, 2, 1, 3],
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false
        }, {
            label: '붉은 선',
            data: [1, -1, 2, 0, 2],
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1,
            fill: false
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
}); 