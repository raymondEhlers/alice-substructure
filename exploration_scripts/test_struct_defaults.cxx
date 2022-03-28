
struct MyStruct {
  int a;
  int b;
};

void test(std::vector<double> values, MyStruct s = MyStruct{2,3})
{
  for (auto & v: values) {
    std::cout << ", " << v << "\n";
  }
  std::cout << s.a << "\n";
}
