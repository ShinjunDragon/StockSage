// script.js
document.addEventListener("DOMContentLoaded", function() {
    const tabs = document.querySelectorAll(".tab");
    const tabContents = document.querySelectorAll(".tab-content");

    tabs.forEach(tab => {
        tab.addEventListener("click", function() {
            const target = this.getAttribute("data-tab");

            tabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(tc => tc.classList.remove("active"));

            this.classList.add("active");
            document.querySelector(`.${target}`).classList.add("active");
        });
    });
});


document.addEventListener('DOMContentLoaded', () => {
  // 모든 .ranking-number 요소를 선택
  const rankingNumbers = document.querySelectorAll('.ranking-number');

  // 각 요소의 스타일을 설정
  rankingNumbers.forEach((element) => {
    element.style.width = '3px';  // 너비 설정
    element.style.height = '10px'; // 높이 설정
    element.style.lineHeight = '10px'; // 텍스트 수직 정렬을 위해 line-height 설정
    element.style.fontSize = '8px'; // 폰트 크기 설정
  });
});

// 주식 코드나 종목명 입력시 리스트 나오게 하는 함수
$(document).ready(function() {
    var searchTimeout;
    $('#stock-search').on('input', function() {
        clearTimeout(searchTimeout);
        var query = $(this).val();
        searchTimeout = setTimeout(function() {
            if (query.length > 0) {
                $.ajax({
                    url: '/stock/search_stocks/',
                    data: {
                        'query': query
                    },
                    success: function(data) {
                        var results = $('#search-results');
                        results.empty();
                        if (data.stocks && data.stocks.length > 0) {
                            var list = $('<ul style="list-style-type: none; padding: 0;">');
                            data.stocks.slice(0, 5).forEach(function(stock) {
                                list.append($('<li style="padding: 5px; cursor: pointer;">').text(stock.code + ' - ' + stock.name)
                                    .click(function() {
                                        $('#stock-search').val(stock.code);
                                        results.empty();
                                    }));
                            });
                            results.append(list);
                        } else {
                            results.text('일치하는 주식이 없습니다.');
                        }
                    },
                    error: function() {
                        $('#search-results').text('검색 중 오류가 발생했습니다.');
                    }
                });
            } else {
                $('#search-results').empty();
            }
        }, 300);
    });
});

$(document).ready(function() {
    var searchTimeout;
    $('#stock-search1').on('input', function() {
        clearTimeout(searchTimeout);
        var query = $(this).val();
        searchTimeout = setTimeout(function() {
            if (query.length > 0) {
                $.ajax({
                    url: '/stock/search_stocks/',
                    data: {
                        'query': query
                    },
                    success: function(data) {
                        var results = $('#search-results1');
                        results.empty();
                        if (data.stocks && data.stocks.length > 0) {
                            var list = $('<ul style="list-style-type: none; padding: 0;">');
                            data.stocks.slice(0, 5).forEach(function(stock) {
                                list.append($('<li style="padding: 5px; cursor: pointer;">').text(stock.code + ' - ' + stock.name)
                                    .click(function() {
                                        $('#stock-search1').val(stock.code);
                                        results.empty();
                                    }));
                            });
                            results.append(list);
                        } else {
                            results.text('일치하는 주식이 없습니다.');
                        }
                    },
                    error: function() {
                        $('#search-results1').text('검색 중 오류가 발생했습니다.');
                    }
                });
            } else {
                $('#search-results1').empty();
            }
        }, 300);
    });
});
