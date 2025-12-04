import React from "react";

//memo설정 안했을때
// export default function B_ChildComponent({ options }) {
//   console.log("Child 렌더링", options);

//   return (
//     <div>
//       <h3>차트 옵션</h3>
//       <p>color: {options.color}</p>
//       <p>type: {options.type}</p>
//     </div>
//   );
// }

// React모듈의 memo()함수를 호출
const B_ChildComponent = React.memo(function B_ChildComponent({ options }) {
  console.log("Child 렌더링", options);

  return (
    <div>
      <h5>차트 옵션</h5>
      <p>color: {options.color}</p>
      <p>type: {options.type}</p>
    </div>
  );
});

export default B_ChildComponent;
