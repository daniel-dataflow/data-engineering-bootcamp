import React, { useState, useMemo } from "react";

export default function B_UseMemoCom() {
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
  //num이 변경될때만 다시 계산하게 설정
  const memoData = useMemo(() => {
    return slowSquare();
  }, [num]);
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
      <p>결과: {memoData}</p>
      <button onClick={toggleTheme}>테마 바꾸기</button>
    </div>
  );
}
