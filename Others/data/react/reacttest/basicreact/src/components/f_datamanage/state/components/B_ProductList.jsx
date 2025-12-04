import React from "react";
export default function ProductList({ products = [], setProducts }) {
  const deleteProduct = (e) => {
    setProducts((prev) => prev.filter((p) => p.productNo != e.target.value));
  };

  return (
    <>
      {products.length > 0 ? (
        <table>
          <thead>
            <tr>
              {Object.keys(products[0]).map((head) => (
                <th key={head}>{head}</th>
              ))}
              <th>비고</th>
            </tr>
          </thead>
          <tbody>
            {products.map((product, i) => (
              <tr key={i}>
                {Object.values(product).map((p, i) => (
                  <td key={i}>{p}</td>
                ))}
                <td>
                  <button onClick={deleteProduct} value={product.productNo}>
                    삭제
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div>
          <h3>조회된 상품이 없습니다.</h3>
        </div>
      )}
    </>
  );
}
