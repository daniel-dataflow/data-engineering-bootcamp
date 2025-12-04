import React, { useState, useMemo } from "react";

const products = [
  { id: 1, name: "노트북", price: 1500000 },
  { id: 2, name: "마우스", price: 20000 },
  { id: 3, name: "키보드", price: 50000 },
  { id: 4, name: "모니터", price: 300000 },
];
export default function B_DataFilterCom() {
  const [minPrice, setMinPrice] = useState(0);
  const [keyword, setKeyword] = useState("");
  const [theme, setTheme] = useState("white");
  // useMemo를 하지 않으면 관련없는 값이 수정되도 리랜더링 되버림
  //   const filteredProducts = () => {
  //     console.log("필터링 + 정렬 계산 실행");
  //     // (1) 최소 가격 기준 필터링
  //     let result = products.filter((p) => p.price >= minPrice);

  //     // (2) 이름에 키워드가 포함된 것만 필터링 (공백이면 전체)
  //     if (keyword.trim() !== "") {
  //       const lower = keyword.toLowerCase();
  //       result = result.filter((p) => p.name.toLowerCase().includes(lower));
  //     }

  //     // (3) 가격 오름차순 정렬
  //     result.sort((a, b) => a.price - b.price);

  //     return result;
  //   };
  const filteredProducts = useMemo(() => {
    console.log("필터링 + 정렬 계산 실행");

    // (1) 최소 가격 기준 필터링
    let result = products.filter((p) => p.price >= minPrice);

    // (2) 이름에 키워드가 포함된 것만 필터링 (공백이면 전체)
    if (keyword.trim() !== "") {
      const lower = keyword.toLowerCase();
      result = result.filter((p) => p.name.toLowerCase().includes(lower));
    }

    // (3) 가격 오름차순 정렬
    result.sort((a, b) => a.price - b.price);

    return result;
  }, [minPrice, keyword]);
  return (
    <div>
      <h4>고정 데이터 리스트 필터하기</h4>
      <h4>상품목록</h4>
      {["white", "black"].map((v) => {
        return (
          <label>
            <input
              type="radio"
              name="theme"
              onClick={(e) => setTheme(e.target.value)}
              value={v}
            />
            {v}
          </label>
        );
      })}
      <div style={{ marginBottom: "1rem", background: theme }}>
        <label>
          최소가격 :{" "}
          <input
            type="number"
            value={minPrice}
            onChange={(e) => setMinPrice(Number(e.target.value))}
          />
        </label>
        <br></br>
        <label>
          상품명 검색 :{" "}
          <input
            type="text"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
          />
        </label>
        <ul>
          {/* {filteredProducts().map((p) => { */}
          {filteredProducts.map((p) => {
            return (
              <li key={p.id}>
                {p.name} - {p.price.toLocaleString()}원
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
