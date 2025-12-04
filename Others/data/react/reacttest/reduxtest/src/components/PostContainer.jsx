import React, { useEffect } from "react";
import { requestPost } from "../features/posts/postSlice";
import { useDispatch, useSelector } from "react-redux";
export default function PostContainer() {
  const { posts, loading, error } = useSelector((state) => state.posts);
  const dispatch = useDispatch();
  useEffect(() => {
    dispatch(requestPost());
  }, [dispatch]);
  if (loading) return <h3>로딩중...</h3>;
  if (error) return <h3>에러 발생 {error}</h3>;
  return (
    <div>
      {posts.slice(0, 3).map((post) => (
        <article key={post.id} style={{ marginBottom: 8 }}>
          <h3>{post.title}</h3>
          <p>{post.body}</p>
        </article>
      ))}
    </div>
  );
}
