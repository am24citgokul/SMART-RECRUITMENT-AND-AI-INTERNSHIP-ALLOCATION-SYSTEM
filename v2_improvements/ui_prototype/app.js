fetch('/api/dashboard')
  .then(function(r) { return r.json(); })
  .then(function(d) {
    var s = d.student;
    document.querySelector('#student-name').textContent = s.name;
    document.querySelector('#skill-index').textContent = s.skill_index;
    var badge = document.querySelector('#skill-level');
    badge.textContent = s.skill_level;
    badge.className = s.skill_level.toLowerCase();
    document.querySelector('#explanation').textContent = s.explanation;

    var items = function(list) {
      return list.map(function(x) { return '<li>' + x.replaceAll('_', ' ') + '</li>'; }).join('');
    };
    document.querySelector('#features').innerHTML = items(s.top_features);

    document.querySelector('#match').textContent = d.match.explanation;
    document.querySelector('#requirements').innerHTML = d.match.requirements.map(function(x) {
      return '<span>' + x.replaceAll('_', ' ') + '</span>';
    }).join('');

    var radar = s.radar;
    new Chart(document.querySelector('#radar'), {
      type: 'radar',
      data: {
        labels: radar.map(function(x) { return x.feature.replaceAll('_', ' '); }),
        datasets: [{
          label: "Relative position in dataset range (%)",
          data: radar.map(function(x) { return x.scaled; }),
          backgroundColor: 'rgba(37,99,235,.18)',
          borderColor: '#2563eb'
        }]
      },
      options: {
        scales: {
          r: {
            beginAtZero: true,
            max: 100
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: function(context) {
                var x = radar[context.dataIndex];
                return x.raw + ' (dataset range ' + x.min + '–' + x.max + '), ' + x.scaled + '%';
              }
            }
          }
        }
      }
    });

    new Chart(document.querySelector('#timeline'), {
      type: 'line',
      data: {
        labels: d.timeline.map(function(x) { return x.label; }),
        datasets: [{
          data: d.timeline.map(function(x) { return x.value; }),
          borderColor: '#16a34a',
          backgroundColor: '#16a34a',
          tension: .35
        }]
      },
      options: {
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { min: 0, max: 100 }
        }
      }
    });
  })
  .catch(function(e) {
    document.body.innerHTML = '<main><h1>Run the V2 pipeline first</h1><p>' + e + '</p></main>';
  });

