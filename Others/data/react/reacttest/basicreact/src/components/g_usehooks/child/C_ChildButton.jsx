import React from "react";

function C_ChildButton({ label, title, onClick }) {
  console.log("C_ChildButtond이 랜더링됨! " + label);
  return (
    <div>
      <button onClick={onClick}>{title}</button>
    </div>
  );
}
export default React.memo(C_ChildButton);
