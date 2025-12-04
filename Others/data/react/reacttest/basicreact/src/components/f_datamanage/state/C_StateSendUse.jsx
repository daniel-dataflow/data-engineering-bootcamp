import React, { useState } from "react";
import ProductList from "./components/B_ProductList";
import ProductInput from "./components/B_ProductInput";
export default function C_StateSendUse() {
  const [products, setProducts] = useState([]);
  return (
    <div>
      <h3>자식에게 state를 전달해서 활용하기</h3>
      <ProductList products={products} setProducts={setProducts} />
      <ProductInput setProducts={setProducts} />
    </div>
  );
}
