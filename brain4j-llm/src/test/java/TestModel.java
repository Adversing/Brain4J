import org.brain4j.llm.Models;
import org.brain4j.llm.core.model.LLM;

public class TestModel {

    public static void main(String[] args) throws Exception {
        LLM llm = Models.loadModel("arnir0/Tiny-LLM");
        System.out.println(llm.id());
        System.out.println(llm.totalSize());
    }
}
