import React, { useState, useMemo } from "react";
import B_ChildComponent from "./B_ChildComponent";

export default function B_ParentComponent() {
  const [color, setColor] = useState("blue");
  const [count, setCount] = useState(0);
  const options = useMemo(() => {
    return {
      color,
      type: "bar",
    };
  }, [color]);
  return (
    <div>
      <h4>부모 컴포넌트</h4>
      <button onClick={() => setCount((c) => c + 1)}>
        count 증가 (부모만 변경) -> 변경하면 자식이 랜더링됨.
      </button>
      <p>count: {count}</p>

      <button
        onClick={() => setColor((prev) => (prev === "blue" ? "red" : "blue"))}
      >
        색상 변경 -> 색상변경시에만 랜더링하려면 useMemo()를 자식에 설정
      </button>

      {/* Child는 options의 참조가 변경될 때만 렌더링 됨 */}
      <B_ChildComponent options={options} />
    </div>
  );
}
