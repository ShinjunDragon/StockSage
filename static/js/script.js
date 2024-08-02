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



