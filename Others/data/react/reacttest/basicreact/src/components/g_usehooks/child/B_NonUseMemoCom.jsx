import React, { useState } from "react";

// state값이 변경될때 마다 페이지가 랜더링 되면서
// 비용 큰 함수를 계속 실행하게 되어 랜더링 속도가 느려짐
export default function B_NonUseMemoCom() {
  const [num, setNum] = useState(1);
  const [theme, setTheme] = useState("light");

  // 1) 느린 계산 함수 분리
  const slowSquare = (n) => {
    console.log("매우 느린 계산 수행 중...(useMemo)");
    let result = n * n;
    // 가짜로 연산량을 늘려서 느린 작업처럼 보이게 함
    for (let i = 0; i < 1_000_000_000; i++) {
      result = result + i;
    }
    return result;
  };

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const style = {
    padding: "1rem",
    backgroundColor: theme === "light" ? "white" : "black",
    color: theme === "light" ? "black" : "white",
  };

  return (
    <div style={style}>
      <h2>느린 계산기</h2>
      <input
        type="number"
        value={num}
        onChange={(e) => setNum(Number(e.target.value))}
      />
      <p>결과: {slowSquare()}</p>
      <button onClick={toggleTheme}>테마 바꾸기</button>
    </div>
  );
}
