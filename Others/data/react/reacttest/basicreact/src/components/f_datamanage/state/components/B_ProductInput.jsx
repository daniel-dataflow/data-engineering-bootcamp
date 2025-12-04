import React, { useState } from "react";
import { NumberGenerator } from "@/data/exportData";

const productNumberGenerator = NumberGenerator("product");
export default function ProductInput({ setProducts }) {
  const [product, setProduct] = useState({
    productNo: "",
    productName: "",
    price: 0,
    type: "",
    color: "",
  });
  const makeProduct = (e) => {
    const { name, value } = e.target;
    setProduct((product) => {
      return { ...product, [name]: value };
    });
  };
  const type = ["식품", "전자기기", "악세사리", "주방용품"];
  const colors = [
    { key: "빨강", value: "red" },
    { key: "파랑", value: "blue" },
    { key: "주황", value: "orange" },
    { key: "회색", value: "gray" },
  ];
  const containerClass = "flex flex-col w-50 space-y-2";
  return (
    <div className={"flex flex-col items-center bg-blue-50 p-5 space-y-5"}>
      <h3>상품입력</h3>
      <div className={containerClass}>
        <input
          type="text"
          name="productName"
          placeholder="상품이름"
          onChange={makeProduct}
          value={product.productName}
        />
        <input
          type="number"
          min="10000"
          step="1000"
          name="price"
          placeholder="상품가격"
          onChange={makeProduct}
          value={product.price}
        />
        <select name="type" onChange={makeProduct} defaultValue="">
          <option disabled value="">
            선택
          </option>
          {type.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        <div className={"grid grid-cols-2"}>
          {colors.map((color) => (
            <label key={color.key}>
              <input
                name="color"
                type="radio"
                value={color.value}
                onChange={makeProduct}
                checked={product.color == color.value}
              />
              <span className={`bg-${color.value}-900`}>{color.key}</span>
            </label>
          ))}
        </div>
        <button
          onClick={(e) => {
            //저장할 product설정
            const productNo = productNumberGenerator.next().value;
            const tempProduct = {
              ...product,
              productNo: productNo,
            };
            //상품 state에 저장
            setProducts((prev) => {
              return [...prev, tempProduct];
            });
            //상품 초기화
            setProduct({
              productNo: "",
              productName: "",
              price: 0,
              type: "",
              color: "",
            });
          }}
        >
          저장
        </button>
      </div>
    </div>
  );
}
