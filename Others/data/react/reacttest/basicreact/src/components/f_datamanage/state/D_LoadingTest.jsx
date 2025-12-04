import React, { useState, useEffect } from "react";
import C_LoadingComponent from "./components/C_LoadingComponent";
import B_ProductList from "./components/B_ProductList";
import { products } from "@/data/exportData";

export default function C_LoadingTest() {
  const [isLoading, setLoading] = useState(true);
  const [productList, setProductList] = useState();
  useEffect(() => {
    setTimeout(() => {
      setProductList(products);
      setLoading(false);
    }, 3000);
  }, []);

  return (
    <div>
      <h3>라이프라이클 함수와 연결하여 loading화면 출력하기</h3>
      {isLoading ? (
        <C_LoadingComponent></C_LoadingComponent>
      ) : (
        <B_ProductList products={productList} setProducts={setProductList} />
      )}
    </div>
  );
}
